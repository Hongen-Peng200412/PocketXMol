"""为小分子 docking 候选计算辅助检查并合成可用排序分数。

``self_ranking`` 使用生成阶段的轨迹坐标置信度，``tuned_ranking`` 使用独立 ranker
对既有 pose 进行一次微扰恢复得到的置信度；二者都再加 ``no_clashes`` 和
``stereo`` 两个 0/1 项。真实 RMSD 从不进入这些可部署分数，只在评测脚本中形成
oracle 对照。
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
# from rdkit import Chem
# from rdkit.Chem import AllChem

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix
from utils.misc import make_config
from utils.docking_aux_scores import calc_clash, prepare_inputs


def make_ranking_score(gen_path):
    """合并候选主索引、辅助检查与可选 tuned confidence。

    ``gen_info.csv`` 每行一个候选，核心键为 ``filename/data_id/i_repeat``，并含
    ``cfd_traj``；``aux_scores.csv`` 含 ``no_clashes/stereo``；可选
    ``tuned_cfd.csv`` 含 ``tuned_cfd``。返回 DataFrame 保留全部原列并新增
    ``self_ranking`` 与可选 ``tuned_ranking``。
    """

    # ``df_gen``：候选主索引 DataFrame，行粒度为单个 pose。
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    # ``df_aux``：碰撞/立体化学辅助结果 DataFrame，行粒度同样为单个 pose。
    df_aux = pd.read_csv(os.path.join(gen_path, 'aux_scores.csv'))
    # ``df_ranking``：按候选文件和复合物左连接；缺失辅助结果保留为 NaN。
    df_ranking = df_gen.merge(df_aux, on=['filename', 'data_id'], how='left')
    # ``path_tuned``：可选独立 ranker 输出 CSV 路径。
    path_tuned = os.path.join(gen_path, 'tuned_cfd.csv')
    if os.path.exists(path_tuned):
        # ``df_cfd``：独立 ranker 的候选级 tuned_cfd 记录。
        df_cfd = pd.read_csv(path_tuned)
        # ``df_ranking``：再按 repeat 共同键左连接，避免同名候选跨轮错位。
        df_ranking = df_ranking.merge(df_cfd, on=['filename', 'data_id', 'i_repeat'], how='left')
    else:
        print('Tuned confidence scores (tuned_cfd.csv) not found.')
        
    # ``self_ranking``：float；原始轨迹置信度加两个布尔检查，各检查通过贡献 1。
    df_ranking['self_ranking'] = df_ranking['cfd_traj'] + df_ranking['no_clashes'].astype('int') + df_ranking['stereo'].astype('int')
    if 'tuned_cfd' in df_ranking.columns:
        # ``tuned_ranking``：float；用 tuned_cfd 替换 cfd_traj，其余两项相同。
        df_ranking['tuned_ranking'] = df_ranking['tuned_cfd'] + df_ranking['no_clashes'].astype('int') + df_ranking['stereo'].astype('int')
    return df_ranking


if __name__ == '__main__':
    # ``parser``：排序脚本命令行解析器。
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='./outputs_test/dock_posebusters')
    parser.add_argument('--db', type=str, default='poseboff')
    parser.add_argument('--sub_dir', type=str, default='')
    parser.add_argument('--gt_mol', type=str, default='')  # set as 'config' for use
    # ``args``：Namespace，保存实验定位、数据集、候选子目录和使用模式开关。
    args = parser.parse_args()

    # ``result_root``：str，多个生成实验所在的根目录。
    result_root = args.result_root
    # ``exp_name``：str/前缀，由 get_dir_from_prefix 解析到唯一实验目录。
    exp_name = args.exp_name
    # ``gen_path``：str，实际生成实验目录。
    gen_path = get_dir_from_prefix(result_root, exp_name)
    # ``save_path``：str，候选辅助检查 CSV 输出路径。
    save_path = os.path.join(gen_path, 'aux_scores.csv')
    # if os.path.exists(save_path):
    #     print(f'Already exists {save_path}, skip')
    #     exit()
    
    if args.gt_mol == '':  # for test set
        if args.db != '':
            # ``db``：str，显式指定测试集名称。
            db = args.db
        else:
            # ``db``：list[str]，先从实验名提取 ``dock_<db>_`` 片段。
            db = re.findall(r'dock_([a-z]+)_', exp_name) 
            if len(db) != 1:
                print('Not found db, use poseboff as default')
                # 提取失败时回退到 poseboff。
                db = 'poseboff'
            else:
                # 唯一匹配时把 list 解包成数据库名 str。
                db = db[0]
        # ``file_dir``：str，测试集参考 mol/protein/pocket 文件共同根目录。
        file_dir = f'data/{db}/files'
        assert os.path.exists(file_dir), f'file_dir {file_dir} does not exist'
    else:
        # ``yml_path``：str，生成目录中保存的任务 YAML 文件名；源码取第一个匹配项。
        yml_path = [f for f in os.listdir(gen_path) if f.endswith('.yml')][0]
        # ``sa_config``：原生成配置，用于取示例模式的 mol/protein 显式路径。
        sa_config = make_config(os.path.join(gen_path, yml_path))
        # ``file_dir``：普通 dict，叶来自 sa_config.data，供 prepare_inputs 的 use 分支读取。
        file_dir = dict(sa_config.data)

    
    # ``df_gen``：候选主索引，每行一个生成 pose。
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))

    # ``sub_dir``：str，候选文件子目录；空 CLI 值回退到标准 SDF。
    sub_dir = 'SDF' if not args.sub_dir else args.sub_dir
    # ``inputs_list``：候选级六叶路径字典列表，顺序与 df_gen 行一致。
    inputs_list = prepare_inputs(df_gen, gen_path, file_dir, sub_dir=sub_dir)
    
    with Pool(64) as p:
        # ``clash_results``：完成顺序可能与输入不同的候选级辅助结果字典列表。
        clash_results = list(tqdm(p.imap_unordered(
            partial(calc_clash),
            inputs_list), total=len(inputs_list)))
    
    # ``df_clash``：辅助结果 DataFrame，通过 filename/data_id 可恢复与主索引的对齐。
    df_clash = pd.DataFrame(clash_results)
    df_clash.to_csv(save_path, index=False)
    
    
    print('Making ranking score: ranking.csv')
    # ``df_ranking``：合并后的完整候选表，含 self/tuned ranking。
    df_ranking = make_ranking_score(gen_path)
    df_ranking.to_csv(os.path.join(gen_path, 'ranking.csv'), index=False)
    
    print('Done')
