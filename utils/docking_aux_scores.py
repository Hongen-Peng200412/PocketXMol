"""为小分子 docking 候选计算排序用的碰撞与立体化学辅助字段。

该模块不计算 docking RMSD，也不等价于完整 PoseBusters PB-valid。它只把每个
``gen_info.csv`` 候选对齐到预测配体、参考配体和蛋白文件，再返回可与模型
confidence 相加的两个布尔检查及碰撞计数。
"""

import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from utils.buster_tools import check_intermolecular_distance, check_identity


def calc_clash(inputs, th=1):
    """检查一个候选的蛋白碰撞与相对参考配体的立体化学身份。

    输入参数:
        - inputs: Mapping，单候选文件与身份路径映射。
        - inputs.data_id: str，复合物/口袋标识，原样写回结果。
        - inputs.filename: str，候选文件名，原样写回结果。
        - inputs.pred_path: str，预测配体 SDF 路径；小分子候选以 ``.sdf`` 结尾。
        - inputs.gt_path: str，参考配体 SDF 路径或 SMILES 文本。
        - inputs.protein_path: str，完整蛋白 PDB 路径，用于分子间距离检查。
        - inputs.pocket_path: str，当前函数读取但不参与计算的兼容字段。
        - th: 任意值，历史兼容参数；当前实现未读取，不改变碰撞阈值。

    返回字段:
        - clash_results.data_id: str，输入复合物标识。
        - clash_results.filename: str，输入候选文件名。
        - clash_results.no_clashes: bool，PoseBusters 距离检查是否没有阈值内碰撞；检查异常时为假。
        - clash_results.num_clashes: int|NaN，碰撞配体—蛋白原子对数量；检查异常时为 NaN。
        - clash_results.rel_clashes: float|NaN，``num_clashes/预测分子原子数``。
        - clash_results.stereo: bool，预测与参考配体的综合 InChI 立体层是否一致；检查异常时为假。

    注意:
        - 小分子传给检查器的 ``clash_cutoff=0.75`` 是 ``distance/(sum_radii*radius_scale)`` 的无量纲阈值，不是 0.75 Å。
        - 条件蛋白侧忽略氢、水、有机辅因子与无机辅因子；预测配体侧固定忽略氢。
    """

    # ``data_id``：str，候选所属复合物标识。
    data_id = inputs['data_id']
    # ``filename``：str，候选文件名，与 gen_info.csv 行键一致。
    filename = inputs['filename']
    # ``pred_path``：str，生成配体文件绝对/相对路径。
    pred_path = inputs['pred_path']
    # ``gt_path``：str，参考配体文件路径或 SMILES 文本。
    gt_path = inputs['gt_path']
    # ``protein_path``：str，受体蛋白 PDB 路径。
    protein_path = inputs['protein_path']
    # ``pocket_path``：str，当前函数不消费，仅保持输入契约完整。
    pocket_path = inputs['pocket_path']
    

    # load pocket
    # ``protein``：未 sanitize 且不按距离自动连键的 RDKit 蛋白 Mol，保留 PDB 坐标。
    protein = Chem.MolFromPDBFile(protein_path, sanitize=False, proximityBonding=False)
    if pred_path.endswith('.sdf'):
        # ``is_pep=False``：小分子候选采用 0.75 的 clash 比例阈值。
        is_pep = False
        # ``mol``：未 sanitize 的预测小分子 RDKit Mol，保留生成 pose 坐标。
        mol = Chem.MolFromMolFile(pred_path, sanitize=False)
    elif pred_path.endswith('.pdb'):
        is_pep = True
        mol = Chem.MolFromPDBFile(pred_path, sanitize=False)

    try:
        # ``clash_results``：PoseBusters 风格嵌套映射，核心叶位于 ``results``。
        clash_results = check_intermolecular_distance(
            mol,
            protein,
            ignore_types={"hydrogens", "organic_cofactors", "inorganic_cofactors", "waters"},
            clash_cutoff=0.75 if not is_pep else 0.65,
        )
        # ``no_clashes``：bool，分子间距离检查是否未发现碰撞。
        no_clashes = clash_results['results']['no_clashes']
        # ``num_clashes``：int，触发阈值的配体—蛋白原子对数量。
        num_clashes = clash_results['results']['num_pairwise_clashes']
    except Exception as e:
        # 检查失败时保守记为有碰撞，而不是丢弃该候选行。
        no_clashes = False
        # ``num_clashes``：float NaN，区分工具失败与真实的 0 个碰撞。
        num_clashes = np.nan
        print(f'Error in check_intermolecular_distance for {data_id} {filename}')
    # ``rel_clashes``：float，每个配体原子的碰撞原子对数；工具失败时保持 NaN。
    rel_clashes = num_clashes / mol.GetNumAtoms()
    
    if gt_path.endswith('.sdf'):
        # ``mol_true``：未 sanitize 的参考小分子 SDF，用于身份/立体化学比较。
        mol_true = Chem.MolFromMolFile(gt_path, sanitize=False)
    elif gt_path.endswith('.pdb'):
        mol_true = Chem.MolFromPDBFile(gt_path, sanitize=False)
    else: # smiles
        # ``mol_true``：从 SMILES 构造的参考二维分子，无三维坐标要求。
        mol_true = Chem.MolFromSmiles(gt_path)
    try:
        # ``tet_results``：身份检查器嵌套结果；字符串 ``w`` 原样透传给 RDKit InChI 模块。
        tet_results = check_identity(
            mol,
            mol_true,
            inchi_options="w",
        )
        # ``stereo``：bool，预测与参考配体的立体化学身份是否通过。
        stereo = tet_results['results']['stereo']
    except Exception as e:
        # 工具异常时保守记为立体检查不通过。
        stereo = False
        print(f'Error in check_identity for {data_id} {filename}')

    # ``clash_results``：扁平候选级结果，供 DataFrame 与 gen_info.csv 合并。
    clash_results = {
        'data_id': data_id,
        'filename': filename,
        'no_clashes': no_clashes,
        'num_clashes': num_clashes,
        'rel_clashes': rel_clashes,
        'stereo': stereo,
    }

    return clash_results


def prepare_inputs(df_gen, gen_dir, file_dir, sub_dir='SDF'):
    """把生成索引的每一行解析成 ``calc_clash`` 所需的路径叶字段。

    输入参数:
        - df_gen: DataFrame，每行一个候选，至少含 ``filename`` 与 ``data_id`` 两列。
        - df_gen.filename: str 列，候选文件名。
        - df_gen.data_id: str 列，复合物/口袋标识。
        - gen_dir: str，生成实验目录。
        - file_dir: str|Mapping，测试集根目录或 use 模式显式路径映射。
        - file_dir.mol_path: str，use 模式的参考/输入配体路径。
        - file_dir.protein_path: str，use 模式的受体 PDB 路径。
        - sub_dir: str，测试集模式下 ``gen_dir`` 内的候选子目录，默认 ``SDF``。

    返回字段:
        - inputs_list: list[dict]，顺序与 ``df_gen.iterrows()`` 一致，每项对应一个候选。
        - inputs_list[i].data_id: str，第 i 个候选的复合物标识。
        - inputs_list[i].filename: str，第 i 个候选的文件名。
        - inputs_list[i].pred_path: str，第 i 个预测配体文件路径。
        - inputs_list[i].gt_path: str，第 i 个参考配体文件路径或 SMILES 文本。
        - inputs_list[i].protein_path: str，第 i 个完整受体蛋白 PDB 路径。
        - inputs_list[i].pocket_path: str，第 i 个 10 Å 口袋路径；use 模式为空字符串，当前 ``calc_clash`` 不读取。
    """

    # ``inputs_list``：按候选行顺序累积的路径契约列表。
    inputs_list = []
    # ``line``：Series，当前候选的 ``filename/data_id`` 等索引字段；``_``：未使用的 DataFrame 行索引。
    for _, line in (df_gen.iterrows()):
        # ``filename``：str，生成候选文件名。
        filename = line['filename']
        # ``data_id``：str，决定测试集参考文件的共同前缀。
        data_id = line['data_id']
        
        if not isinstance(file_dir, dict):  # test set
            # ``pred_path``：测试集生成子目录中的候选文件路径。
            pred_path = os.path.join(gen_dir, sub_dir, filename)
            if not os.path.exists(pred_path):
                print(f'pred_path {pred_path} not exist. Are you using openmm as sub_dir? Use SDF for this case!')
                # 指定子目录缺失时回退到标准 SDF 子目录。
                pred_path = os.path.join(gen_dir, 'SDF', filename)
            if filename.endswith('.sdf'):
                # ``gt_path``：小分子参考 SDF，命名为 ``<data_id>_mol.sdf``。
                gt_path = os.path.join(file_dir, 'mols', data_id+'_mol.sdf')
            elif filename.endswith('.pdb'):
                gt_path = os.path.join(file_dir, 'peptides', data_id+'_pep.pdb')
            # ``protein_path``：完整受体蛋白 PDB。
            protein_path = os.path.join(file_dir, 'proteins', data_id+'_pro.pdb')
            # ``pocket_path``：10 Å 口袋 PDB；当前 calc_clash 不消费，但保留给兼容调用。
            pocket_path = os.path.join(file_dir, 'pockets10', data_id+'_pocket.pdb')
        else:  # use
            # ``pred_path``：示例使用模式的 ``<实验名>_SDF`` 输出目录。
            pred_path = os.path.join(gen_dir, os.path.basename(gen_dir)+'_SDF', filename)
            # ``gt_path``：调用者显式提供的输入/参考配体路径。
            gt_path = file_dir['mol_path']
            # ``protein_path``：调用者显式提供的蛋白 PDB 路径。
            protein_path = file_dir['protein_path']
            # 使用模式没有独立 pocket 文件，以空字符串占位。
            pocket_path = ''
        
    
            # continue
        inputs_list.append({
            'data_id': data_id,
            'filename': filename,
            'pred_path': pred_path,
            'gt_path': gt_path,
            'protein_path': protein_path,
            'pocket_path': pocket_path,
        })
    return inputs_list
    
    
