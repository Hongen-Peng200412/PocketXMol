"""按构象生成基准协议计算分子级 Coverage 与 Matching。

对每个二维分子图建立参考构象 × 生成构象 RMSD 矩阵。RMSD 使用 RDKit
``GetBestRMS``，允许刚体对齐并枚举对称原子映射；因此它衡量分子内部构象差异，
与 docking 中不允许移动预测 pose 的 RMSD 定义不同。GEOM-Drug 默认要求生成数
恰为参考构象数的两倍，并使用 1.25 Å Coverage 阈值。
"""

import pandas as pd
import numpy as np
import os
import copy
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem import rdMolAlign as MA
from multiprocessing import Pool
import argparse
import re
import json
import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix

def load_gt(data_path):
    """载入并按非手性规范 SMILES 聚合参考构象。

    输入参数:
        - data_path: str，参考 pickle 路径；表中每行一个参考 conformer。
        - pickle.smi: str 列，原始分子 SMILES 分组键。
        - pickle.mol: RDKit Mol 列，带三维坐标的参考 conformer。

    返回值:
        - gt_dict: dict[str, list[RDKit Mol]]，键为第一个参考分子重算的非手性规范 SMILES，值为同一二维图的去氢参考构象列表。
    """

    # ``data``：DataFrame，初始每行一个参考构象。
    data = pd.read_pickle(data_path)
    # ``data``：每行一个输入 smi，mol 列聚合为参考 RDKit Mol 列表。
    data = data.groupby("smi")["mol"].apply(list).reset_index()
    # ``data['mol']``：list[RDKit Mol]，逐构象移除显式氢以匹配生成 SDF 的重原子协议。
    data['mol'] = data['mol'].apply(lambda x: [Chem.RemoveAllHs(mol) for mol in x])
    # ``data['smi']``：str，用第一个参考分子重算非手性规范 SMILES 作为稳定分组键。
    data['smi'] = data['mol'].apply(lambda x: Chem.MolToSmiles(x[0], isomericSmiles=False))

    # ``gt_dict``：非手性 SMILES -> 参考构象 Mol 列表。
    gt_dict = data[['smi', 'mol']].set_index('smi').to_dict()['mol']
    return gt_dict


def load_gen(gen_path, test_df_path):
    """把生成目录中的 SDF 按其测试集二维图聚合成候选构象集合。

    输入参数:
        - gen_path: str，生成实验目录；读取其 ``gen_info.csv`` 与 ``SDF`` 子目录。
        - test_df_path: str，测试集索引 CSV 路径。
        - test_csv.data_id: str 列，生成索引与测试二维图之间的连接键。
        - test_csv.smiles: str 列，生成构象的二维图分组键。
        - gen_info.data_id: str 列，每个候选所属测试分子标识。
        - gen_info.filename: str 列，每个候选在 ``SDF`` 子目录中的文件名。

    返回值:
        - gen_dict: dict[str, list[RDKit Mol|None]]，键沿用测试表 SMILES，值按 ``gen_info.csv`` 行顺序保存已存在的候选 SDF 解析结果。

    注意:
        - 缺失的 SDF 文件被跳过；RDKit 解析失败得到的 None 仍会进入列表。
        - 测试表 SMILES 后续必须能与 ``load_gt`` 重算的非手性规范键一致。
    """

    # ``df_test``：测试分子索引 DataFrame。
    df_test = pd.read_csv(test_df_path)
    # ``data_id_to_smi``：data_id -> SMILES 的一对一映射。
    data_id_to_smi = df_test[['data_id', 'smiles']].set_index('data_id').to_dict()['smiles']

    # ``df_gen``：候选主索引，每行一个已生成构象文件。
    df_gen = pd.read_csv(os.path.join(gen_path, "gen_info.csv"))
    # ``sdf_dir``：标准候选 SDF 子目录。
    sdf_dir = os.path.join(gen_path, "SDF")

    # ``gen_dict``：SMILES -> 生成构象 Mol 列表，按 gen_info.csv 行顺序追加。
    gen_dict = {}
    # ``_``：int，未使用的 DataFrame 原始行索引。
    # ``line``：Series，当前生成候选的文件定位与样本标识列。
    for _, line in tqdm(df_gen.iterrows(), total=len(df_gen), desc='load gen...'):
        # ``data_id``：当前候选所属测试分子标识。
        data_id = line['data_id']
        # ``filename``：当前候选 SDF 文件名。
        filename = line['filename']
        # ``path``：当前候选 SDF 完整路径。
        path = os.path.join(sdf_dir, filename)
        if not os.path.exists(path):
            continue
        # ``mol``：从 SDF 读取并默认 sanitize 的 RDKit Mol。
        mol = Chem.MolFromMolFile(path)
        
        # ``smi``：当前 data_id 的二维图分组键。
        smi = data_id_to_smi[data_id]
        if smi in gen_dict:
            gen_dict[smi].append(mol)
        else:
            # ``gen_dict``：首次出现该二维图时创建只含当前生成构象的列表。
            gen_dict[smi] = [mol]
    
    return gen_dict


def get_rmsd_min(ref_mols, gen_mols, use_ff=False, threshold=0.5):
    """计算单个二维图的 Coverage 与 Matching。

    输入参数:
        - ref_mols: list[RDKit Mol]，长度为 R，同一二维图的参考 conformer。
        - gen_mols: list[RDKit Mol]，长度为 G，同一二维图的生成 conformer。
        - use_ff: bool，为真时先对每个生成构象副本执行 MMFF 局部优化。
        - threshold: float，Coverage 成功阈值，单位 Å。

    返回值:
        - cov: float，R 个参考构象中、最近生成 RMSD 不超过 ``threshold`` 的比例。
        - mat: float，R 个参考构象各自最近生成 RMSD 的均值，单位 Å。
    """

    # ``rmsd_mat``：Float32 ndarray，形状为 (R, G) Å；行是参考构象，列是生成构象。
    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    # ``i``：int，当前生成构象在 ``rmsd_mat`` 中的列索引。
    # ``gen_mol``：RDKit Mol，第 i 个生成 conformer。
    for i, gen_mol in enumerate(gen_mols):
        # ``gen_mol_c``：当前生成构象的深拷贝，力场优化不会修改原列表对象。
        gen_mol_c = copy.deepcopy(gen_mol)
        if use_ff:
            MMFFOptimizeMolecule(gen_mol_c)
        # ``j``：int，当前参考构象在 ``rmsd_mat`` 中的行索引。
        # ``ref_mol``：RDKit Mol，第 j 个参考 conformer。
        for j, ref_mol in enumerate(ref_mols):
            # ``ref_mol_c``：当前参考构象的深拷贝，供 RDKit 对齐函数安全修改坐标。
            ref_mol_c = copy.deepcopy(ref_mol)
            # ``rmsd_mat``：单元 ``[j,i]`` 写入参考 j 与生成 i 的最优对齐/对称 RMSD（Å）。
            rmsd_mat[j, i] = get_best_rmsd(gen_mol_c, ref_mol_c)
    # ``rmsd_mat_min``：Float32 ndarray，形状为 (R,) Å；每个参考构象到所有生成构象的最近 RMSD。
    rmsd_mat_min = rmsd_mat.min(-1)
    return (rmsd_mat_min <= threshold).mean(), rmsd_mat_min.mean()


def get_best_rmsd(gen_mol, ref_mol):
    """计算允许刚体对齐与对称原子映射的重原子构象 RMSD。

    输入参数:
        - gen_mol: RDKit Mol，生成 conformer；必须与参考分子具有兼容二维拓扑。
        - ref_mol: RDKit Mol，参考 conformer。

    返回值:
        - rmsd: float，允许适当刚体旋转/平移与对称原子映射后的最小重原子 RMSD，单位 Å。

    注意:
        - 与 docking ``CalcRMS`` 不同，``GetBestRMS`` 会移动 probe 分子以最优对齐；它不会把固定手性分子的镜像反射视为一般刚体旋转。
    """

    # ``gen_mol``：去除显式氢后的生成 RDKit Mol。
    gen_mol = Chem.RemoveAllHs(gen_mol)
    # ``ref_mol``：去除显式氢后的参考 RDKit Mol。
    ref_mol = Chem.RemoveAllHs(ref_mol)
    # ``rmsd``：float Å，对称映射与刚体对齐下的最小重原子 RMSD。
    rmsd = MA.GetBestRMS(gen_mol, ref_mol)
    return rmsd


def set_rdmol_positions(rdkit_mol, pos):
    """在参考拓扑副本上按原子索引写入坐标。

    输入参数:
        - rdkit_mol: RDKit Mol，提供固定二维拓扑与 conformer 容器。
        - pos: ndarray，形状为 (N, 3)，按原子索引对齐的生成坐标，单位 Å。

    返回值:
        - mol: RDKit Mol，输入分子的独立去氢副本，conformer 0 坐标已逐原子替换。

    注意:
        - 现有断言在移除显式氢之前比较原子数；默认 ``cal_metrics`` 没有启用该辅助修复路径。
    """

    # ``mol``：输入 RDKit Mol 的深拷贝。
    mol = copy.deepcopy(rdkit_mol)
    assert mol.GetConformer(0).GetPositions().shape[0] == pos.shape[0]
    # ``mol``：移除显式氢后的参考拓扑副本。
    mol = Chem.RemoveAllHs(mol)
    # ``i``：int，0-based 原子索引，与 ``pos`` 的第 0 维及 RDKit 原子顺序共同对齐。
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def print_results(cov, mat):
    """汇总跨分子的 Coverage/Matching 均值与中位数并打印。

    输入参数:
        - cov: Sequence[float]，逐二维图 Coverage，无量纲。
        - mat: Sequence[float]，逐二维图 Matching，单位 Å。

    返回字段:
        - metric_dict.cov_mean: NumPy 标量，跨二维图 Coverage 均值。
        - metric_dict.cov_median: NumPy 标量，跨二维图 Coverage 中位数。
        - metric_dict.mat_mean: NumPy 标量，跨二维图 Matching 均值，单位 Å。
        - metric_dict.mat_median: NumPy 标量，跨二维图 Matching 中位数，单位 Å。
    """

    # ``cov_mean``：NumPy 浮点标量，跨二维图 Coverage 的算术均值，无量纲。
    # ``cov_median``：NumPy 浮点标量，跨二维图 Coverage 的中位数，无量纲。
    cov_mean, cov_median = np.mean(cov), np.median(cov)
    print("COV_mean: ", cov_mean, ";COV_median: ", cov_median)
    # ``mat_mean``：NumPy 浮点标量，跨二维图 Matching 的算术均值，单位 Å。
    # ``mat_median``：NumPy 浮点标量，跨二维图 Matching 的中位数，单位 Å。
    mat_mean, mat_median = np.mean(mat), np.median(mat)
    print("MAT_mean: ", mat_mean, ";MAT_median: ", mat_median)
    return {
        'cov_mean': cov_mean,
        'cov_median': cov_median,
        'mat_mean': mat_mean,
        'mat_median': mat_median,
    }


def single_process(content):
    """解包一个二维分子的构象评测任务并计算两个分子级指标。

    输入参数:
        - content: tuple，长度为 4，元素顺序固定为 ``(ref_mols, gen_mols, use_ff, threshold)``。
        - content[0]: list[RDKit Mol]，同一二维图的 R 个参考构象。
        - content[1]: list[RDKit Mol]，同一二维图的 G 个生成构象。
        - content[2]: bool，是否先对生成构象副本执行 MMFF 局部优化。
        - content[3]: float，Coverage 的 RMSD 成功阈值，单位 Å。

    返回值:
        - cov: float，R 个参考构象中至少被一个生成构象覆盖的比例，无量纲。
        - mat: float，每个参考构象到生成集合最近 RMSD 的均值，单位 Å。
    """

    # ``ref_mols``：list[RDKit Mol]，同一二维图的参考构象集合。
    # ``gen_mols``：list[RDKit Mol]，同一二维图的生成构象集合。
    # ``use_ff``：bool，是否先对生成构象副本执行 MMFF 局部优化。
    # ``threshold``：float，Coverage 判定使用的 RMSD 阈值，单位 Å。
    ref_mols, gen_mols, use_ff, threshold = content
    # ``cov``：float，当前二维图的参考构象 Coverage，无量纲。
    # ``mat``：float，当前二维图的平均最近生成 RMSD，单位 Å。
    cov, mat = get_rmsd_min(ref_mols, gen_mols, use_ff, threshold)
    return cov, mat


def process(content):
    """为多进程评测提供单分子容错包装。

    输入参数:
        - content: tuple，字段顺序与 ``single_process`` 完全相同。

    返回值:
        - result: tuple[float, float]|None，成功时为 ``(cov, mat)``，任意异常时为 None 并由汇总循环跳过该二维图。

    注意:
        - 现有实现吞掉所有异常且不记录失败原因；本学习分支只说明行为，不改变异常策略。
    """

    try:
        return single_process(content)
    except:
        return None


def fix_inconsistency(gt_dict, gen_dict):
    """用参考拓扑承载生成坐标，修复 RDKit 原子属性不一致。

    输入参数:
        - gt_dict: dict[str, list[RDKit Mol]]，非手性 SMILES 到 R 个参考构象的映射。
        - gen_dict: dict[str, list[RDKit Mol]]，相同 SMILES 到 G 个生成构象的映射；列表元素会被原地替换。

    返回值:
        - gen_dict: dict[str, list[RDKit Mol]]，保持键和列表顺序不变，每个生成构象改为“首个参考二维拓扑 + 原生成坐标”。

    注意:
        - 当前 ``cal_metrics`` 注释掉了本函数调用，因此正式主评测路径不执行该修复。
        - 元素逐索引断言只核对元素符号，不验证键、形式电荷或手性标签完全一致。
    """

    # ``smi``：str，当前二维分子的非手性规范 SMILES，同时索引两个输入映射。
    for smi in tqdm(gen_dict.keys(), desc='fix inconsistency'):
        # ``gen_mols``：当前 SMILES 的生成构象列表引用。
        gen_mols = gen_dict[smi]
        # ``ref_mols``：当前 SMILES 的参考构象列表。
        ref_mols = gt_dict[smi]
        
        # if (Chem.MolToSmiles(gen_mols[0], isomericSmiles=False) != 
        #     Chem.MolToSmiles(ref_mols[0], isomericSmiles=False)):
        #     print('fix inconsistency for', smi)
        if True:
            # ``ref_mol``：作为统一二维拓扑模板的第一个参考构象。
            ref_mol = ref_mols[0]
            # ``i``：int，当前生成构象在 ``gen_mols`` 中的索引。
            # ``gen_mol``：RDKit Mol，待按参考拓扑重建的当前生成 conformer。
            for i, gen_mol in enumerate(gen_mols):
                # ``gen_mol``：去除显式氢后的当前生成构象。
                gen_mol = Chem.RemoveAllHs(gen_mol)
                assert all([gen_mol.GetAtomWithIdx(idx).GetSymbol() == \
                            ref_mol.GetAtomWithIdx(idx).GetSymbol() 
                        for idx in range(ref_mol.GetNumAtoms())])
                # ``conf``：ndarray，形状为 (N, 3) Å，当前生成 conformer 坐标。
                conf = gen_mol.GetConformer(0).GetPositions()
                # ``new_mol``：参考拓扑副本，坐标替换为当前生成构象。
                new_mol = set_rdmol_positions(ref_mol, conf)
                # ``gen_mols``：用统一拓扑的新分子替换当前生成列表位置，保持构象顺序不变。
                gen_mols[i] = new_mol
        else:
            continue
    return gen_dict


def cal_metrics(gen_path, data_path, 
                test_df_path, use_ff, threshold):
    """载入全数据集、并行计算逐分子 COV/MAT，并返回明细表。

    输入参数:
        - gen_path: str，生成实验目录；``load_gen`` 从中读取 ``gen_info.csv`` 与 ``SDF``。
        - data_path: str，参考构象 pickle 路径。
        - test_df_path: str，测试二维图 ``data_id -> smiles`` 映射 CSV 路径。
        - use_ff: bool，是否在 RMSD 前对每个生成构象副本执行 MMFF 局部优化。
        - threshold: float，Coverage 的 RMSD 成功阈值，单位 Å。

    返回值:
        - metric_dict: dict[str, NumPy scalar]，跨二维分子的总体构象指标。
        - metric_dict.cov_mean: NumPy scalar，逐分子 Coverage 均值，无量纲。
        - metric_dict.cov_median: NumPy scalar，逐分子 Coverage 中位数，无量纲。
        - metric_dict.mat_mean: NumPy scalar，逐分子 Matching 均值，单位 Å。
        - metric_dict.mat_median: NumPy scalar，逐分子 Matching 中位数，单位 Å。
        - df: DataFrame，索引为成功 worker 输入在 ``content_list`` 中的 0-based 序号。
        - df.cov: float 列，对应二维图的 Coverage，无量纲。
        - df.mat: float 列，对应二维图的 Matching，单位 Å。

    异常:
        - ValueError: 任一二维图的生成构象数不等于其参考构象数的两倍。
    """

    # ``gt_dict``：非手性 SMILES -> 参考构象列表。
    gt_dict = load_gt(data_path)
    # ``gen_dict``：测试表 SMILES -> 生成构象列表。
    gen_dict = load_gen(gen_path, test_df_path)
    
    # gen_dict = fix_inconsistency(gt_dict, gen_dict)
    
    print('num of covered mol graphs of gt', len(set(gt_dict.keys()).intersection(set(gen_dict.keys()))),
          'out of', len(gt_dict.keys()))

    print('num of mol confs in gt:', sum([len(x) for x in gt_dict.values()]))
    print('num of mol confs in gen:', sum([len(x) for x in gen_dict.values()]))
    
    # check num of mol graphs
    # assert len(set(gt_dict.keys()).intersection(set(gen_dict.keys()))) == 188

    # ``cov_list``：list[float]，按成功 worker 结果顺序累积分子级 Coverage。
    # ``mat_list``：list[float]，按成功 worker 结果顺序累积分子级 Matching，单位 Å。
    cov_list, mat_list = [], []
    # ``index_list``：成功结果在 content_list 中的原序号，用作明细 DataFrame 索引。
    index_list = []
    # ``content_list``：每个二维图一个四元组 worker 输入。
    content_list = []
    # ``smi``：str，当前二维图的非手性规范 SMILES，同时索引参考与生成构象映射。
    for smi in gen_dict.keys():
        # ``ref_mols``：当前二维图的 R 个参考构象。
        ref_mols = gt_dict[smi]
        # ``gen_mols``：当前二维图的 G 个生成构象。
        gen_mols = gen_dict[smi]
        if len(gen_mols) != 2 * len(ref_mols):
            # print('Warning: num of generated mols is not twice of num of ref mols')
            raise ValueError('num of generated mols is not twice of num of ref mols')
        content_list.append((ref_mols, gen_mols, use_ff, threshold))

    # ``pool``：含 64 个 worker 的多进程池，按 ``content_list`` 顺序执行容错包装 ``process``。
    with Pool(64) as pool:
        # ``index``：int，当前 worker 输入在 ``content_list`` 中的序号。
        # ``inner_output``：tuple[float, float]|None，当前二维图的 Coverage、Matching 返回对或失败标记。
        for index, inner_output in tqdm(enumerate(pool.imap(process, content_list))):
            if inner_output is None:
                continue
            # ``cov``：float，当前 worker 返回的分子级 Coverage，无量纲。
            # ``mat``：float，当前 worker 返回的分子级 Matching，单位 Å。
            cov, mat = inner_output
            cov_list.append(cov)
            mat_list.append(mat)
            index_list.append(index)
    # ``metric_dict``：跨分子的 cov/mat mean/median 四叶汇总。
    metric_dict = print_results(cov_list, mat_list)

    # ``df``：逐分子成功结果表，列 cov（无量纲）与 mat（Å）。
    df = pd.DataFrame(zip(cov_list, mat_list), index=index_list, columns=['cov', 'mat'])
    return metric_dict, df


if __name__ == '__main__':
    # ``parser``：构象评测命令行解析器。
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='geom')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--result_root', type=str, default='./outputs')
    # ``args``：Namespace，含数据库名、生成实验前缀与结果根目录。
    args = parser.parse_args()
    
    # ``gen_path``：str，由结果根目录和实验前缀解析出的唯一生成目录。
    gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    print('Evaluate conf in', gen_path)

    # ``data_path``：临时 lambda，把数据子集名映射到参考 pickle 路径。
    data_path = lambda x: f'data/test/conf/rdkit_cluster_data/{x}/test_data_200.pkl'
    if args.db_name == 'geom':
        # ``data_path``：str，GEOM-Drug 的聚类参考构象 pickle。
        data_path = data_path('drugs')
        # ``threshold``：float Å，GEOM-Drug Coverage 成功阈值。
        threshold = 1.25
    elif args.db_name == 'qm9':
        # ``data_path``：str，QM9 参考构象 pickle；下行主动阻止未核实阈值的评测。
        data_path = data_path('qm9')
        raise NotImplementedError('check threshold for qm9')
        # ``threshold``：不可达的历史占位值，单位 Å。
        threshold = 0.5
    else:
        raise ValueError(f'Unknown db_name: {args.db_name}')
    # ``test_df_path``：str，data_id 到 SMILES 的测试集映射 CSV。
    test_df_path = f'data/test/dfs/conf_{args.db_name}.csv'
    # ``use_uff``：bool；False 表示评测生成坐标本身，不先做 MMFF 优化。
    use_uff = False

    # ``metric_dict``：dict[str, NumPy 标量]，总体 Coverage 与 Matching 的均值和中位数四叶汇总。
    # ``df``：DataFrame，逐分子成功结果的 ``cov`` 与 ``mat`` 明细表。
    metric_dict, df = cal_metrics(gen_path, data_path,
                        test_df_path, use_uff, threshold)
                    
    # save
    with open(os.path.join(gen_path, 'metric.txt'), 'w') as f:
        json.dump({k:str(v) for k,v in metric_dict.items()}, f, indent=2)
    df.to_csv(os.path.join(gen_path, 'df_metric.csv'))
