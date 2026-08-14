"""计算小分子 docking pose RMSD，并评估可用排序与候选池上限。

受体口袋坐标系已经固定，因此 ``get_rmsd`` 使用不会移动预测分子的 RDKit
``CalcRMS``；配体整体平移/旋转属于 docking 误差。默认成功阈值为 2 Å。脚本还
把 ``ranking.csv`` 的 self/tuned 排序与事后 oracle 并列，但 PB-valid 需另由
``evaluate_by_buster.py`` 计算，不能从 RMSD 成功率推断。
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import re

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix


def get_rmsd(mol_prob, mol_gt):
    """计算不移动预测 pose、但考虑对称原子匹配的重原子 RMSD。

    输入参数:
        - mol_prob: RDKit Mol，预测小分子 pose，坐标必须处于受体固定坐标系。
        - mol_gt: RDKit Mol，实验小分子 pose，二维拓扑需与预测分子兼容。

    返回值:
        - rmsd: float，不移动预测 pose、但枚举对称原子映射的重原子 RMSD，单位 Å。

    注意:
        - 正常分支最多枚举 30000 个子结构映射；触发 RuntimeError 时退回相同原子索引一一匹配，不再处理对称置换。
        - 两个输入先深拷贝，RDKit 内部操作不会污染调用方对象。
    """
    # mol_prob, mol_gt = mol_pair
    # ``mol_prob``：预测 pose 的独立 RDKit Mol 副本。
    mol_prob = deepcopy(mol_prob)
    # ``mol_gt``：实验 pose 的独立 RDKit Mol 副本。
    mol_gt = deepcopy(mol_gt)
    try:
        # ``rmsd``：float Å；CalcRMS 枚举匹配但不刚体移动 probe。
        rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, maxMatches=30000)  # NOT move mol_prob. for docking
    except RuntimeError:
        # ``n_atoms``：int，预测分子的原子数；回退映射假定两分子索引完全对齐。
        n_atoms = mol_prob.GetNumAtoms()
        # ``map_list``：单个 identity atom map，形如 [[(0, 0),...,(N-1, N-1)]]。
        map_list = [[(i, i) for i in range(n_atoms)]]
        # ``rmsd``：identity 映射下的不移动 RMSD，单位 Å。
        rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, map=map_list)
    # rmsd = Chem.rdMolAlign.GetBestRMS(mol_prob, mol_gt)  # move mol_prob to mol_gt
    return rmsd


def set_rdmol_positions(rdkit_mol, pos):
    """在去氢后的参考拓扑副本上按原子索引写入预测坐标。

    输入参数:
        - rdkit_mol: RDKit Mol，提供参考二维拓扑与 conformer 容器。
        - pos: ndarray，形状为 (N, 3)，与去氢后原子索引对齐的预测坐标，单位 Å。

    返回值:
        - mol: RDKit Mol，去氢参考拓扑的独立副本，conformer 0 坐标已逐原子替换。

    注意:
        - 该函数用于原子属性不一致修复路径，当前 ``evaluate_rmsd_df`` 中对应调用被注释。
    """

    # ``rdkit_mol``：移除显式氢后的拓扑对象；此调用可能返回新 Mol。
    rdkit_mol = Chem.RemoveAllHs(rdkit_mol)
    assert rdkit_mol.GetConformer(0).GetPositions().shape[0] == pos.shape[0]
    # ``mol``：去氢拓扑的深拷贝，后续逐原子写坐标。
    mol = deepcopy(rdkit_mol)
    # ``i``：int，0-based 原子索引，与 ``pos`` 的第 0 维及 RDKit 原子顺序共同对齐。
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def fix_inconsistency_one_mol(mol_prob, mol_gt):
    """把预测坐标装到实验拓扑上，以统一原子属性。

    输入参数:
        - mol_prob: RDKit Mol，提供预测 pose 坐标。
        - mol_gt: RDKit Mol，提供实验二维拓扑。

    返回值:
        - new_mol: RDKit Mol，实验拓扑的独立副本，坐标替换为预测 pose。

    异常:
        - AssertionError: 两分子相同原子索引处的元素符号不一致。

    注意:
        - 默认 RMSD 主链没有启用该辅助修复。
    """

    assert all([mol_gt.GetAtomWithIdx(idx).GetSymbol() == \
                mol_prob.GetAtomWithIdx(idx).GetSymbol() 
            for idx in range(mol_gt.GetNumAtoms())])
    # ``conf``：ndarray，形状为 (N, 3) Å，预测 pose 的 conformer 坐标。
    conf = mol_prob.GetConformer(0).GetPositions()
    # ``new_mol``：实验拓扑副本，坐标替换为预测 pose。
    new_mol = set_rdmol_positions(mol_gt, conf)
    return new_mol


def evaluate_rmsd_df(df_gen, gen_dir, gt_dir, check_repeats=10):
    """为候选主索引逐行追加实验 pose RMSD。

    输入参数:
        - df_gen: DataFrame，每行一个候选，至少含 ``data_id`` 与 ``filename`` 两列。
        - df_gen.data_id: str 列，候选所属复合物标识。
        - df_gen.filename: str 列，候选 SDF 文件名。
        - df_gen.i_repeat: int 列，可选，采样轮编号；本函数不读取。
        - gen_dir: str，实际候选 SDF 所在目录。
        - gt_dir: str，参考配体目录；文件命名为 ``<data_id>_mol.sdf``。
        - check_repeats: int，大于 0 时只断言 ``总候选数/唯一 data_id 数`` 等于该值，不逐组检查每个 data_id 的候选数。

    返回字段:
        - df_gen: DataFrame，原对象，索引被原地重置并新增 ``rmsd`` 列。
        - df_gen.rmsd: float 列，不移动预测 pose 的对称 RMSD，单位 Å；文件缺失、解析失败或空分子行保留 NaN。

    注意:
        - ``check_repeats`` 的现有断言只验证全局平均数，不能保证每个复合物恰有相同候选数；本学习分支只记录而不修复。
    """

    # ``data_id_list``：ndarray，形状为 (M,)，候选表中唯一复合物标识。
    data_id_list = df_gen['data_id'].unique()
    print('Find %d generated mols with %d unique data_id' % (len(df_gen), len(data_id_list)))
    if check_repeats > 0:
        assert len(df_gen) / len(data_id_list) == check_repeats, f'Repeat {check_repeats} not match: {len(df_gen)}:{len(data_id_list)}'

    # ``gt_files``：data_id -> 实验配体 SDF 路径。
    gt_files = {data_id: os.path.join(gt_dir, data_id+'_mol.sdf')
                for data_id in data_id_list}
    # ``gt_mols``：data_id -> 默认 sanitize 后的实验 RDKit Mol。
    gt_mols = {data_id: Chem.MolFromMolFile(gt_files[data_id])
               for data_id in data_id_list}

    # ``df_gen['rmsd']``：float 列，先填 NaN，再逐候选写入 Å 值。
    df_gen['rmsd'] = np.nan
    df_gen.reset_index(inplace=True, drop=True)
    # ``index``：int，重置后的候选行索引，同时定位 ``df_gen.rmsd`` 的写入行。
    # ``line``：Series，当前候选的文件名、样本标识和已有评分列。
    for index, line in tqdm(df_gen.iterrows(), total=len(df_gen)):
        # if index % len(data_id_list) == 28:
        #     continue
        # ``data_id``：当前候选所属复合物标识。
        data_id = line['data_id']
        # ``filename``：当前候选 SDF 文件名。
        filename = line['filename']
        # ``file_path``：当前候选 SDF 完整路径。
        file_path = os.path.join(gen_dir, filename)
        if not os.path.exists(file_path):
            print('Warning: Not found %s' % filename)
            continue
        # ``gen_mol``：默认 sanitize 读取的预测 pose RDKit Mol。
        gen_mol = Chem.MolFromMolFile(file_path)
        if gen_mol is None:
            # ``gen_mol``：sanitize 失败时重读坐标与拓扑但关闭 sanitize。
            gen_mol = Chem.MolFromMolFile(os.path.join(gen_dir, filename), sanitize=False)
        if gen_mol is None:
            print(f'Error mol: {filename}')
            continue
        if gen_mol.GetNumAtoms() == 0:
            print('Warning: Empty mol: %s' % filename)
            continue
        # gen_mol = fix_inconsistency_one_mol(gen_mol, gt_mols[data_id])
        # ``rmsd``：float Å，不移动预测 pose 的对称 RMSD。
        rmsd = get_rmsd(gen_mol, gt_mols[data_id])
        # ``df_gen.loc``：按重置后的行索引把当前 RMSD 写回候选表对应行。
        df_gen.loc[index, 'rmsd'] = rmsd
        
        # confidence
        # if os.path.exists(os.path.join(gen_dir, filename.replace('.sdf', '.pt'))):
        #     output = torch.load(os.path.join(gen_dir, filename.replace('.sdf', '.pt')))
        #     df_gen.loc[index, 'confidence'] = torch.mean(output['confidence_pos']).item()
    
    return df_gen

    
def get_topk_metrics(df_rmsd, topk):
    """评估 confidence 前 k 候选池中按真实 RMSD 排序的完整名次。

    输入参数:
        - df_rmsd: DataFrame，每行一个 docking 候选，至少含 ``data_id`` 与 ``rmsd``，可选含 ``confidence``。
        - df_rmsd.data_id: str 列，候选所属复合物标识。
        - df_rmsd.rmsd: float 列，不移动预测 pose 的 RMSD，单位 Å。
        - df_rmsd.confidence: float 列，可选，值越大排序越靠前；缺失时函数原地新增 ``-rmsd`` 作为 oracle 分数。
        - topk: int，每个复合物先按 confidence 保留的候选数量 k。

    返回值:
        - df_topk_rmsd: DataFrame，每个 ``data_id`` 一行，含一个身份列与 k 个真实 RMSD 名次列。
        - df_topk_rmsd.data_id: str 列，复合物标识。
        - df_topk_rmsd.rank0..rank{k-1}: float 列，confidence 前 k 候选再按真实 RMSD 升序排列的值，单位 Å；rank0 是该候选池的 oracle 最优值。
        - oracle: bool，输入缺少 confidence 并使用 ``-rmsd`` 代替时为真。

    注意:
        - 返回名次衡量 confidence 前 k 候选池的事后上限，不表示排序器自动选中了最低 RMSD pose。
    """

    # ``oracle``：bool，记录是否因缺少 confidence 而泄漏真实 RMSD 构造排序。
    oracle = False
    if 'confidence' not in df_rmsd.columns:
        # use real rmsd as -confidence
        # ``df_rmsd.confidence``：新列 ``confidence=-rmsd`` 使升序 RMSD 等价于降序 confidence，仅作 oracle。
        df_rmsd['confidence'] = -df_rmsd['rmsd']
        # ``oracle``：标记返回表不是可部署排序结果。
        oracle = True

    # get rmsd from confidence topk in each data_id and explode
    # ``df_topk_rmsd``：先按 data_id 分组，保留 confidence 降序前 k 行。
    df_topk_rmsd = df_rmsd.groupby('data_id').apply(
        lambda x: x.sort_values('confidence', ascending=False).head(topk)
    ).reset_index(drop=True)
    # ``df_topk_rmsd``：每组 rmsd 转为升序 list[float] Å。
    df_topk_rmsd = df_topk_rmsd.groupby('data_id')['rmsd'].apply(
        lambda x: x.sort_values(ascending=True).tolist()).reset_index()
    # ``df_topk_rmsd``：展开 list 为 rank0..rank{k-1} 列，rank0 是候选池内 oracle 最优值。
    df_topk_rmsd[[f'rank{i}' for i in range(topk)]] = pd.DataFrame(df_topk_rmsd['rmsd'].tolist())
    # ``df_topk_rmsd``：删除中间 list 列，只保留复合物键与展开名次列。
    df_topk_rmsd = df_topk_rmsd.drop(columns=['rmsd'])
    
    return df_topk_rmsd, oracle


def get_rank_metrics(df_rmsd, ranks):
    """报告 confidence 前 k 个候选中的最小真实 RMSD。

    输入参数:
        - df_rmsd: DataFrame，每行一个 docking 候选，至少含 ``data_id`` 与 ``rmsd``，可选含 ``confidence``。
        - df_rmsd.data_id: str 列，候选所属复合物标识。
        - df_rmsd.rmsd: float 列，不移动预测 pose 的 RMSD，单位 Å。
        - df_rmsd.confidence: float 列，可选，值越大排序越靠前；缺失时在独立副本中生成随机正态分数。
        - ranks: list[int]，需要评估的正整数候选池大小；列表顺序决定新增 ``rank_k`` 列的写入顺序。

    返回值:
        - df_metric: DataFrame，每个 ``data_id`` 一行，保存全池 oracle 与各 confidence 候选池的最优真实 RMSD。
        - df_metric.data_id: str 列，复合物标识。
        - df_metric.best: float 列，该复合物全部候选中的最小真实 RMSD，单位 Å。
        - df_metric.rank_<k>: float 列，confidence 前 k 个候选中的最小真实 RMSD，单位 Å。
        - cfd: bool，输入确实含 confidence 列时为真；使用随机排序回退时为假。

    注意:
        - 缺失 confidence 时的随机回退不是 oracle，且受 NumPy 全局随机状态影响。
    """

    # ``df_rmsd``：独立副本，避免新增 confidence 列污染调用者。
    df_rmsd = df_rmsd.copy()
    if 'confidence' not in df_rmsd.columns:
        # use real rmsd as -confidence
        # ``df_rmsd.confidence``：缺失 confidence 时为每个候选生成随机排序分数。
        df_rmsd['confidence'] = np.random.randn(df_rmsd['rmsd'].shape[0])
        # ``cfd``：``cfd=False`` 表示 rank_k 不是模型置信度排序结果。
        cfd = False
    else:
        # ``cfd``：``cfd=True`` 表示使用输入 confidence 列排序。
        cfd = True

    # get the oracle best
    # ``df_metric``：以 data_id 为索引、每组真实 RMSD 最小值为 best 的表。
    df_metric = df_rmsd.groupby('data_id')[['rmsd']].min()
    # ``df_metric.columns``：单列改名为 ``best``，明确它是全候选池事后 oracle。
    df_metric.columns = ['best']

    # get rmsd from confidence topk in each data_id and explode
    # ``df_rank_rmsd``：每个 data_id 内按 confidence 降序排列后重新组成的候选表。
    df_rank_rmsd = df_rmsd.groupby('data_id').apply(
        lambda x: x.sort_values('confidence', ascending=False)
    ).reset_index(drop=True)
    
    # get confidence-based rank top k
    # ``rank``：int，当前 confidence 候选池大小 k，用于切片每组排序后的前 k 行。
    for rank in ranks:
        # ``df_rank``：每个复合物 confidence 前 rank 个候选中的最小 RMSD，单位 Å。
        df_rank = df_rank_rmsd.groupby('data_id')['rmsd'].apply(
            lambda x: x.values[:rank].min()).reset_index(drop=True)
        # ``df_metric``：新列 ``rank_k`` 按当前 groupby 的 data_id 顺序写入对应 Å 值。
        df_metric['rank_%d' % rank] = df_rank.values
    # ``df_metric``：把 data_id 从索引恢复为普通列，便于 CSV/merge。
    df_metric = df_metric.reset_index()
    return df_metric, cfd


if __name__ == '__main__':
    # ``parser``：docking RMSD/排序评测命令行解析器。
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='./outputs_test/dock_posebusters')
    parser.add_argument('--check_repeats', type=int, default=0)
    parser.add_argument('--db', type=str, default='')
    parser.add_argument('--sub_dir', type=str, default='')
    parser.add_argument('--use_repeats', type=int, default=100)
    # ``args``：Namespace，含实验定位、候选子目录、数据集与使用 repeat 上限。
    args = parser.parse_args()

    # ``result_root``：str，生成实验根目录。
    result_root = args.result_root
    # ``exp_name``：str/前缀，用于解析实际实验目录。
    exp_name = args.exp_name
    
    # db = exp_name.split('_')[-2]
    if args.db != '':
        # ``db``：str，显式指定的 docking 测试集名称。
        db = args.db
    else:
        # ``db``：list[str]，先从实验名的 dock_<db>_ 片段提取。
        db = re.findall(r'dock_([a-z]+)_', exp_name)
        if len(db) != 1:
            print('Not found db, use poseboff as default')
            # ``db``：提取失败时回退 poseboff。
            db = 'poseboff'
        else:
            # ``db``：唯一匹配时解包为 str。
            db = db[0]
        assert db in ['pbdock', 'poseb', 'poseboff'], f'Unknown db {db} for docking eval'
    # ``gt_dir``：str，实验配体 SDF 目录；文件名按 data_id 构造。
    gt_dir = f'data/{db}/files/mols'
    assert os.path.exists(gt_dir), f'gt_dir {gt_dir} does not exist'

    # ``gen_path``：str，实际生成实验目录。
    gen_path = get_dir_from_prefix(result_root, exp_name)
    
    # ``df_gen``：候选主索引 DataFrame。
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    
    # # make rmsd df
    if not args.sub_dir:
        # ``rmsd_path``：标准 SDF 候选的 RMSD CSV 输出路径。
        rmsd_path = os.path.join(gen_path, 'rmsd.csv')
        # ``sdf_path``：标准候选 SDF 目录。
        sdf_path = os.path.join(gen_path, 'SDF')
    else:
        # ``rmsd_path``：后处理候选子目录对应的独立 RMSD CSV 路径。
        rmsd_path = os.path.join(gen_path, f'rmsd_{args.sub_dir}.csv')
        # ``sdf_path``：用户指定后处理候选目录。
        sdf_path = os.path.join(gen_path, args.sub_dir)
    # if not os.path.exists(rmsd_path):
    # ``df_gen``：追加 rmsd(Å) 列后的候选表。
    df_gen = evaluate_rmsd_df(df_gen, sdf_path, gt_dir, args.check_repeats)
    df_gen.to_csv(rmsd_path, index=False)
    # else:
    #     df_gen = pd.read_csv(rmsd_path)
    
    
    # rank1 with 
    if os.path.exists(os.path.join(gen_path, 'ranking.csv')):
        # ``df_ranking``：可用 self/tuned 候选排序表，不含真实 RMSD。
        df_ranking = pd.read_csv(os.path.join(gen_path, 'ranking.csv'))
        # ``df_rmsd``：用于 merge 的候选共同键与真实 RMSD 子表。
        df_rmsd = df_gen[['filename', 'data_id', 'i_repeat', 'rmsd']]
        # ``df_ranking``：按三个共同键左连接真实 RMSD，仅用于离线评测。
        df_ranking = df_ranking.merge(df_rmsd, on=['filename', 'data_id', 'i_repeat'], how='left')
        # ``df_ranking``：只保留 i_repeat 小于评测预算的候选池。
        df_ranking = df_ranking[df_ranking['i_repeat'] < args.use_repeats]
        # ``df_rank1``：每个 data_id 一行的 self/tuned 首选 RMSD 与 oracle 最小 RMSD。
        df_rank1 = df_ranking.groupby('data_id').apply(lambda x:
            pd.Series({
                'rmsd_self_ranking': x.sort_values('self_ranking', ascending=False)['rmsd'].iloc[0],
                'rmsd_tuned_ranking': x.sort_values('tuned_ranking', ascending=False)['rmsd'].iloc[0]\
                    if 'tuned_ranking' in df_ranking.columns else np.nan,
                'rmsd_oracle_ranking': x['rmsd'].min(),
            }))
        # df_rank1 = df_rank1.merge(df_gen, on=['data_id'], how='left')
        df_rank1.to_csv(os.path.join(gen_path, 'rank1_rmsd.csv'), index=False)
        
        print('Ratio of RMSD < 2A:')
        print((df_rank1< 2).mean(0))

    print('Done')
