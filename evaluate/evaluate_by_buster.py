"""对小分子 docking 候选运行 PoseBusters ``redock`` 完整有效性检查。

脚本硬编码读取生成目录中的 ``rank1_rmsd_bel.csv``；当前仓库未检索到负责生成该
文件名的可执行代码，因此排序结果必须由外部步骤筛选/改名后再交给本入口。每个预测
SDF 与实验配体、受体蛋白共同送入 PoseBusters，输出 CSV 保留工具返回的所有检查
叶字段，可用于计算 PB-valid；它比排序阶段的 ``no_clashes`` 轻量检查覆盖更广。
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import re
from multiprocessing import Pool
from functools import partial

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from posebusters import PoseBusters

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix



def calc_buster_one(inputs, buster):
    """运行一个候选的 PoseBusters redock 检查并补回候选键。

    输入参数:
        - inputs: Mapping，单候选路径与身份映射。
        - inputs.data_id: str，候选所属复合物标识。
        - inputs.filename: str，候选 SDF 文件名。
        - inputs.pred_path: str，预测小分子 SDF 路径。
        - inputs.gt_path: str，实验小分子 SDF 路径。
        - inputs.protein_path: str，完整受体蛋白 PDB 路径。
        - buster: PoseBusters，以 ``redock`` 检查集合实例化的评测器。

    返回字段:
        - buster_result.data_id: str，输入复合物标识。
        - buster_result.filename: str，输入候选文件名。
        - buster_result.<check>: 标量，PoseBusters ``redock`` 当前版本返回的其余检查叶；动态字段集合由外部包配置决定。

    注意:
        - ``buster.bust`` 返回 DataFrame，本函数只取重置索引后的首行并展平为字典。
    """

    # ``data_id``：str，候选所属复合物标识。
    data_id = inputs['data_id']
    # ``filename``：str，候选文件名。
    filename = inputs['filename']
    # ``pred_path``：str，预测小分子 SDF 路径。
    pred_path = inputs['pred_path']
    # ``gt_path``：str，实验配体 SDF 路径。
    gt_path = inputs['gt_path']
    # ``protein_path``：str，受体蛋白 PDB 路径。
    protein_path = inputs['protein_path']
    
    # ``buster_result``：PoseBusters 返回的候选级 DataFrame。
    buster_result = buster.bust([pred_path], gt_path, protein_path)
    # ``buster_result``：首行转成的检查叶字段字典。
    buster_result = buster_result.reset_index().iloc[0].to_dict()

    buster_result.update({
        # ``buster_result.data_id``：str，候选所属复合物标识，用于与 RMSD/排序表连接。
        'data_id': data_id,
        # ``buster_result.filename``：str，候选 SDF 文件名，用于定位原始生成文件。
        'filename': filename
    })
    return buster_result


def prepare_inputs(df_gen, gen_dir, file_dir, sub_dir='SDF'):
    """把候选索引解析为 PoseBusters 所需三类结构文件路径。

    输入参数:
        - df_gen: DataFrame，每行一个已选候选，至少含 ``filename`` 与 ``data_id`` 两列。
        - df_gen.filename: str 列，候选文件名。
        - df_gen.data_id: str 列，复合物标识。
        - gen_dir: str，生成实验目录。
        - file_dir: str，测试集 ``files`` 根目录，内部应含 ``mols`` 与 ``proteins``。
        - sub_dir: str，候选所在子目录，默认 ``SDF``。

    返回字段:
        - inputs_list: list[dict]，顺序与 DataFrame 行顺序一致。
        - inputs_list[i].data_id: str，第 i 个候选的复合物标识。
        - inputs_list[i].filename: str，第 i 个候选的文件名。
        - inputs_list[i].pred_path: str，第 i 个预测 SDF 路径；指定子目录缺失时回退 ``SDF``。
        - inputs_list[i].gt_path: str，第 i 个实验配体 SDF 路径。
        - inputs_list[i].protein_path: str，第 i 个完整受体蛋白 PDB 路径。
    """
    
    # ``inputs_list``：按 DataFrame 行顺序累积的 PoseBusters 输入列表。
    inputs_list = []
    # ``_``：int，未使用的 DataFrame 原始行索引。
    # ``line``：Series，当前候选的文件定位与样本标识列。
    for _, line in (df_gen.iterrows()):
        # ``filename``：str，候选文件名。
        filename = line['filename']
        # ``data_id``：str，参考配体/蛋白文件共同前缀。
        data_id = line['data_id']
        
        # ``pred_path``：str，优先指向用户指定候选子目录。
        pred_path = os.path.join(gen_dir, sub_dir, filename)
        # ``gt_path``：str，实验小分子 SDF 路径。
        gt_path = os.path.join(file_dir, 'mols', data_id+'_mol.sdf')
        # ``protein_path``：str，完整受体蛋白 PDB 路径。
        protein_path = os.path.join(file_dir, 'proteins', data_id+'_pro.pdb')
        
        if not os.path.exists(pred_path):
            print(f'pred_path {pred_path} not exist. Are you using openmm as sub_dir? Use SDF for this case!')
            # ``pred_path``：指定后处理目录缺文件时回退到原始 SDF 候选。
            pred_path = os.path.join(gen_dir, 'SDF', filename)
            # continue
        # ``inputs_list[i]``：dict，当前候选的身份叶与 PoseBusters 三类结构文件路径。
        inputs_list.append({
            # ``inputs_list[i].data_id``：str，当前候选所属复合物标识。
            'data_id': data_id,
            # ``inputs_list[i].filename``：str，当前候选 SDF 文件名。
            'filename': filename,
            # ``inputs_list[i].pred_path``：str，预测配体 SDF 路径，可能已回退到标准 ``SDF`` 目录。
            'pred_path': pred_path,
            # ``inputs_list[i].gt_path``：str，实验配体 SDF 路径。
            'gt_path': gt_path,
            # ``inputs_list[i].protein_path``：str，完整受体蛋白 PDB 路径。
            'protein_path': protein_path,
        })
    return inputs_list
    

if __name__ == '__main__':
    # ``parser``：PoseBusters 评测命令行解析器。
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='./outputs_test')
    parser.add_argument('--db', type=str, default='')
    parser.add_argument('--sub_dir', type=str, default='')
    # ``args``：Namespace，含实验定位、数据库与候选子目录。
    args = parser.parse_args()

    # ``result_root``：str，生成实验根目录。
    result_root = args.result_root
    # ``exp_name``：str/前缀，用于解析实际实验目录。
    exp_name = args.exp_name
    
    if args.db != '':
        # ``db``：str，显式测试集名称。
        db = args.db
    else:
        # ``db``：list[str]，先从实验名提取 dock_<db>_ 片段。
        db = re.findall(r'dock_([a-z]+)_', exp_name) 
        if len(db) != 1:
            print('Not found db, use poseboff as default')
            # ``db``：提取失败时回退 poseboff。
            db = 'poseboff'
        else:
            # ``db``：唯一匹配时解包为 str。
            db = db[0]
        assert db in ['pbdock', 'poseb', 'poseboff'], f'Unknown db {db} for docking eval'
    # ``file_dir``：str，测试集 mols/proteins 等文件的共同根目录。
    file_dir = f'data/{db}/files'
    assert os.path.exists(file_dir), f'file_dir {file_dir} does not exist'

    # ``gen_path``：str，实际生成实验目录。
    gen_path = get_dir_from_prefix(result_root, exp_name)
    
    # # load gen df
    # ``df_gen``：DataFrame，当前脚本硬编码读取的 rank1 候选索引表；仓库内未找到该文件名的生产入口，需外部准备。
    df_gen = pd.read_csv(os.path.join(gen_path, 'rank1_rmsd_bel.csv'))
    
    # ``sub_dir``：str，候选子目录；空 CLI 值回退 SDF。
    sub_dir = 'SDF' if not args.sub_dir else args.sub_dir
    # ``inputs_list``：候选级五叶 PoseBusters 输入字典列表。
    inputs_list = prepare_inputs(df_gen, gen_path, file_dir, sub_dir=sub_dir)
    
    # ``buster``：使用官方 redock 检查集合的 PoseBusters 实例。
    buster = PoseBusters(config='redock')
    with Pool(40) as p:
        # ``buster_results``：无序并行返回的候选级检查字典列表。
        buster_results = list(tqdm(p.imap_unordered(
            partial(calc_buster_one, buster=buster),
            inputs_list), total=len(inputs_list)))
    
    # ``df_buster``：每行一个候选、列为全部 PoseBusters 检查叶的 DataFrame。
    df_buster = pd.DataFrame(buster_results)
    df_buster.to_csv(os.path.join(gen_path, f'buster{args.sub_dir}.csv'), index=False)
    
    print('Done')
