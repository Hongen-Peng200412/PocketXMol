"""执行原去噪采样循环, 将批级状态、输出和轨迹拆回单分子.

先读sample_loop3: 每步由noiser重新加噪, 模型预测干净变量, outputs2batch再写回下一步状态. 可选progress只记录Python调用阶段与前向进入/返回次数, 不参与科学计算. seperate_outputs2负责逐图拆分, get_cfd_traj负责原轨迹置信度聚合.
本模块不落盘. sample_loop3返回(batch, outputs, trajs), 核心字段见函数Docstring. B为候选图数, N和H为批内配体原子数与完整无向半边数, T为一轮步数; trajs各坐标/类别数组的首轴为时间, outputs.confidence_pos_traj则为(N,T).
"""

# Standard library imports
import os
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

# ``ALT_KEYS.node``：str，轨迹短名 ``node`` 对应 Batch 当前原子类别叶 ``node_type``。
# ``ALT_KEYS.pos``：str，轨迹短名 ``pos`` 对应 Batch 当前原子坐标叶 ``node_pos``。
# ``ALT_KEYS.halfedge``：str，轨迹短名 ``halfedge`` 对应 Batch 当前完整半边类别叶 ``halfedge_type``。
ALT_KEYS = {
    'node': 'node_type',
    'pos': 'node_pos',
    'halfedge': 'halfedge_type',
}


def add_info_to_dict(data_dict, data, mode):
    """把一个阶段的三个分子变量追加到轨迹容器。

    输入参数:
        - data_dict: MutableMapping，必须预建 ``node``、``pos``、``halfedge`` 三个列表叶。
        - data_dict.node: list[Tensor]，每项形状为 (N,)，保存某阶段的原子类别索引。
        - data_dict.pos: list[Tensor]，每项形状为 (N, 3)，保存某阶段的原子坐标，单位 Å。
        - data_dict.halfedge: list[Tensor]，每项形状为 (H,)，保存某阶段的完整半边类别索引。
        - data: Mapping|PyG Batch，读取叶由 ``mode`` 决定，实体顺序必须与当前批次一致。
        - data.node_in: Tensor，可选，形状为 (N,)，``mode=in`` 时读取的带噪原子类别。
        - data.pos_in: Tensor，可选，形状为 (N, 3)，``mode=in`` 时读取的带噪坐标，单位 Å。
        - data.halfedge_in: Tensor，可选，形状为 (H,)，``mode=in`` 时读取的带噪半边类别。
        - data.node_type: Tensor，可选，形状为 (N,)，``mode=out`` 时读取的投影后原子类别。
        - data.node_pos: Tensor，可选，形状为 (N, 3)，``mode=out`` 时读取的投影后坐标，单位 Å。
        - data.halfedge_type: Tensor，可选，形状为 (H,)，``mode=out`` 时读取的投影后半边类别。
        - data.gt_node_type: Tensor，可选，形状为 (N,)，``mode=gt`` 时读取的干净原子类别参照。
        - data.gt_node_pos: Tensor，可选，形状为 (N, 3)，``mode=gt`` 时读取的干净坐标参照，单位 Å。
        - data.gt_halfedge_type: Tensor，可选，形状为 (H,)，``mode=gt`` 时读取的干净半边类别参照。
        - data.pred_node: Tensor，可选，形状为 (N, C_n)，``mode=raw_out`` 时读取的原子类别 logits。
        - data.pred_pos: Tensor，可选，形状为 (N, 3)，``mode=raw_out`` 时读取的原始坐标预测，单位 Å。
        - data.pred_halfedge: Tensor，可选，形状为 (H, C_e)，``mode=raw_out`` 时读取的半边类别 logits。
        - mode: Literal[in,out,gt,raw_out]，指定上述读取分支。

    返回值:
        - None: 原地向 ``data_dict`` 的三个列表叶追加 CPU Tensor。

    注意:
        - 所有保存项都会 ``detach().cpu()``，因此轨迹不持有计算图或 GPU 显存。
        - ``gt_*`` 字段缺失时跳过对应变量，而不是补空张量。
    """
    assert mode in ['in', 'out', 'gt', 'raw_out'], f'Unknown mode: {mode}'
    # ``key``：str，当前处理的轨迹短名，依次为 ``node``、``pos``、``halfedge``。
    for key in ['node', 'pos', 'halfedge']:
        if mode == 'in':
            # ``this``：Tensor，当前 ``key`` 的带噪输入；三种形状依次为 (N,)、(N, 3)、(H,)。
            this = data[key+'_in']
        elif mode == 'out':
            # ``this``：M-Projector 写回后的当前离散/坐标状态。
            this = data[ALT_KEYS[key]]
        elif mode == 'gt':
            if 'gt_' + ALT_KEYS[key] in data:
                # ``this``：采样开始前保留的干净参照，与对应实体顺序对齐。
                this = data['gt_' + ALT_KEYS[key]]
            else:
                continue
        else:  # raw_out
            if key != 'pos':
                # ``this``：Tensor，形状为 (N,) 或 (H,)，仅为轨迹可视化保存主预测 argmax，不覆盖 outputs。
                this = data['pred_' + key].argmax(dim=-1)
            else:
                # ``this``：Tensor，形状为 (N, 3)，坐标输出本身已是连续干净变量预测，单位 Å。
                this = data['pred_' + key]
        data_dict[key].append(this.detach().cpu())


def sample_loop3(batch, model, noiser, device=None, is_ar='', off_tqdm=False, progress=None):
    """执行"重新加噪-直接去噪-任务投影"的迭代采样.

    输入参数:
        - batch: PyG Batch, 已完成分子/口袋 featurizer 与任务 transform, 含 B 个候选图.
        - batch.node_type: LongTensor, 形状为 (N,), 当前原子类别.
        - batch.node_pos: FloatTensor, 形状为 (N, 3), 当前配体坐标, 单位 Å.
        - batch.halfedge_type: LongTensor, 形状为 (H,), 当前完整无向半边类别.
        - batch.gt_node_type: LongTensor, 形状为 (N,), 采样开始前保存的干净原子类别参照.
        - batch.gt_node_pos: FloatTensor, 形状为 (N, 3), 历史调用可保存干净坐标参照; 本项目docking传全零轨迹占位, 不是沉积真值.
        - batch.gt_halfedge_type: LongTensor, 形状为 (H,), 采样开始前保存的干净半边类别参照.
        - batch.fixed_node: LongTensor|BoolTensor, 形状为 (N,), 原子类别 prompt; 0 表示待恢复, 1 表示条件.
        - batch.fixed_pos: LongTensor|BoolTensor, 形状为 (N,), 坐标 prompt; 0 表示待恢复, 1 表示条件.
        - batch.fixed_halfedge: LongTensor|BoolTensor, 形状为 (H,), 半边类别 prompt; 0 表示待恢复, 1 表示条件.
        - batch.fixed_halfdist: LongTensor|BoolTensor, 形状为 (H,), 半边距离 prompt; 0 表示待恢复, 1 表示条件.
        - model: PMAsymDenoiser, 把写入 ``*_in`` 的 Batch 映射为干净变量和 confidence 预测.
        - noiser: ConfSampleNoiser|DockSamplNoiser, 提供 ``steps_loop``、``__call__``、``outputs2batch`` 与 ``num_steps``.
        - device: torch.device|str|None, 辅助批次需要迁移时使用的设备.
        - off_tqdm: bool, 是否关闭进度条, 不影响采样数值.
        - is_ar: str, 空字符串是本项目普通单轮采样; 原ar/ar2分支按已有规则追加自回归步骤, 本项目不启用.
        - progress: dict|None, 可选原地诊断字典, None保持旧调用; 不修改张量、随机数或采样规则.
            - progress.stage: str, 当前Python执行阶段, prepare_loop、noise、forward、prediction_to_batch或trajectory.
            - progress.model_forward_attempt_count: int, 调用方初始设0, 每次进入批量模型调用前增加1.
            - progress.model_forward_completed_count: int, 调用方初始设0, 每次模型Python调用完整返回后增加1; CUDA异步错误可能之后才观察到.

    返回值采用当前无linking分块、非自回归的普通单轮路径: N为批内配体原子总数, H为完全图半边总数, num_node_types和num_edge_types为模型词表宽度. T_total=num_steps, S_all=1+2*num_steps; 正式100步时分别为100和201. 所有预测坐标均相对pocket_center对应的模型局部原点.
        - batch: PyG Batch, 最后一次 ``outputs2batch`` 投影后的最终状态.
        - batch.node_type: LongTensor, 形状为 (N,), 最终原子类别.
        - batch.node_pos: FloatTensor, 形状为 (N, 3), 最终生成坐标, 单位 Å.
        - batch.halfedge_type: LongTensor, 形状为 (H,), 最终半边类别.
        - outputs: dict, 最后一步模型输出, 并新增坐标 confidence 轨迹叶.
        - outputs.pred_node: FloatTensor, 形状为 (N, num_node_types), 最后一步原子类别 logits.
        - outputs.pred_pos: FloatTensor, 形状为 (N, 3), 最后一步干净坐标预测, 单位 Å.
        - outputs.pred_halfedge: FloatTensor, 形状为 (H, num_edge_types), 最后一步半边类别 logits.
        - outputs.confidence_node: FloatTensor, 形状为 (N, 1), 最后一步原子 confidence 原始输出.
        - outputs.confidence_pos: FloatTensor, 形状为 (N, 1), 最后一步坐标 confidence 原始输出.
        - outputs.confidence_halfedge: FloatTensor, 形状为 (H, 1), 最后一步半边 confidence 原始输出.
        - outputs.confidence_pos_traj: FloatTensor, 形状为 (N, T_total), 逐原子逐步坐标 confidence 原始输出.
        - trajs: dict, 顶层叶为 ``all``、``in``、``out``、``raw`` 四种轨迹来源.
        - trajs.all.node: ndarray, 形状为 (S_all, N), 依次交错保存 gt、每步输入和每步投影后的原子类别.
        - trajs.all.pos: ndarray, 形状为 (S_all, N, 3), 依次交错保存gt_node_pos、每步输入和每步投影后的坐标, 单位Å; 当前docking首项为全零占位.
        - trajs.all.halfedge: ndarray, 形状为 (S_all, H), 依次交错保存 gt、每步输入和每步投影后的半边类别.
        - trajs.in.node: ndarray, 形状为 (T_total, N), 每步带噪原子类别.
        - trajs.in.pos: ndarray, 形状为 (T_total, N, 3), 每步带噪坐标, 单位 Å.
        - trajs.in.halfedge: ndarray, 形状为 (T_total, H), 每步带噪半边类别.
        - trajs.out.node: ndarray, 形状为 (T_total, N), 每步投影后原子类别.
        - trajs.out.pos: ndarray, 形状为 (T_total, N, 3), 每步投影后坐标, 单位 Å.
        - trajs.out.halfedge: ndarray, 形状为 (T_total, H), 每步投影后半边类别.
        - trajs.raw.node: ndarray, 形状为 (T_total, N), 模型原始原子类别 argmax.
        - trajs.raw.pos: ndarray, 形状为 (T_total, N, 3), 模型原始坐标预测, 单位 Å.
        - trajs.raw.halfedge: ndarray, 形状为 (T_total, H), 模型原始半边类别 argmax.
    """
    if progress is not None:
        progress['stage'] = 'prepare_loop'
    # ``traj_dict``: dict[str, list[Tensor]], all 轨迹容器; 三个叶先放 gt, 随后交错追加 in/out.
    traj_dict = {
        # ``traj_dict.node``: list[Tensor], 每项形状为 (N,), 保存一个阶段的原子类别索引.
        'node': [],
        # ``traj_dict.pos``: list[Tensor], 每项形状为 (N, 3), 保存一个阶段的配体坐标, 单位 Å.
        'pos': [],
        # ``traj_dict.halfedge``: list[Tensor], 每项形状为 (H,), 保存一个阶段的完整半边类别索引.
        'halfedge': [],
    }
    # ``cfd_traj``: list[Tensor], 每项形状为 (N,), 按采样步保存逐原子坐标置信度原始值.
    cfd_traj = []
    mol_parts = get_mol_parts_linking(batch, device=device)
    for data_mol in [batch] + mol_parts:
        add_info_to_dict(traj_dict, data_mol, 'gt')

    # ``in_dict``: dict[str, list[Tensor]], 仅保存每步 noiser 产生的带噪变量.
    in_dict = {
        # ``in_dict.node``: list[Tensor], 每项形状为 (N,), 逐步带噪原子类别索引.
        'node': [],
        # ``in_dict.pos``: list[Tensor], 每项形状为 (N, 3), 逐步带噪配体坐标, 单位 Å.
        'pos': [],
        # ``in_dict.halfedge``: list[Tensor], 每项形状为 (H,), 逐步带噪完整半边类别索引.
        'halfedge': [],
    }
    # ``out_dict``: dict[str, list[Tensor]], 仅保存每步 M-Projector 更新后的当前状态.
    out_dict = {
        # ``out_dict.node``: list[Tensor], 每项形状为 (N,), 逐步投影后的原子类别索引.
        'node': [],
        # ``out_dict.pos``: list[Tensor], 每项形状为 (N, 3), 逐步投影后的配体坐标, 单位 Å.
        'pos': [],
        # ``out_dict.halfedge``: list[Tensor], 每项形状为 (H,), 逐步投影后的完整半边类别索引.
        'halfedge': [],
    }
    # ``raw_dict``: dict[str, list[Tensor]], 仅保存模型原始干净预测; 类别 logits 转成 argmax 类别.
    raw_dict = {
        # ``raw_dict.node``: list[Tensor], 每项形状为 (N,), 模型原始原子类别 logits 的 argmax 索引.
        'node': [],
        # ``raw_dict.pos``: list[Tensor], 每项形状为 (N, 3), 模型原始坐标预测, 单位 Å.
        'pos': [],
        # ``raw_dict.halfedge``: list[Tensor], 每项形状为 (H,), 模型原始半边类别 logits 的 argmax 索引.
        'halfedge': [],
    }
    
    step_ar = 0
    while True:
        # ``step``: float, init_step*k/num_steps的采样进度; 正式100步依次为1.0, 0.99, ..., 0.01, 再由任务level scaler映射为信息等级.
        for step in tqdm(noiser.steps_loop(add_last=False), desc='Sampling steps', 
                        total=noiser.num_steps, disable=off_tqdm):
            with torch.no_grad():
                if progress is not None:
                    progress['stage'] = 'noise'
                # ``batch``: 在当前状态上按 step 对应 level 重新加噪, 并写入 ``*_in``.
                batch = noiser(batch, step)
                if progress is not None:
                    progress['stage'] = 'trajectory'
                add_info_to_dict(in_dict, batch, 'in')
                add_info_to_dict(traj_dict, batch, 'in')
                
                # ``outputs``: 模型干净变量与置信度预测; 节点/半边顺序仍与 Batch 对齐.
                if progress is not None:
                    progress['stage'] = 'forward'
                    progress['model_forward_attempt_count'] += 1
                outputs = model(batch) 
                if progress is not None:
                    progress['model_forward_completed_count'] += 1
                    progress['stage'] = 'prediction_to_batch'
                batch.update({'step': step})
                # ``batch``: M-Projector 后的下一步当前状态; free 路径直接采用主预测.
                batch = noiser.outputs2batch(batch, outputs)  # M-Projector (what a strange name, I (xingang) do not like it)
                if progress is not None:
                    progress['stage'] = 'trajectory'
                add_info_to_dict(out_dict, batch, 'out')
                add_info_to_dict(traj_dict, batch, 'out')
                add_info_to_dict(raw_dict, outputs, 'raw_out')
                cfd_traj.append(outputs['confidence_pos'].detach().flatten())

        if is_ar.startswith('ar'):
            if is_ar == 'ar':
                batch = noiser.outputs2batch_ar(batch, outputs, step_ar, cfd_traj)
            elif is_ar == 'ar2':
                batch = noiser.outputs2batch_ar2(batch, outputs)
            add_info_to_dict(out_dict, batch, 'out')
            add_info_to_dict(traj_dict, batch, 'out')
            add_info_to_dict(raw_dict, outputs, 'raw_out')
            if batch['node_p2'].shape[0] == 0:
                break
            step_ar += 1
        else:
            break
    
    # 四个字典的叶最终都把时间/阶段轴堆到第 0 维.
    # ``all_trajs``: dict[str, ndarray], 稍后把 all 轨迹三个叶沿阶段轴堆叠.
    # ``in_trajs``: dict[str, ndarray], 稍后把输入轨迹三个叶沿时间轴堆叠.
    # ``out_trajs``: dict[str, ndarray], 稍后把投影轨迹三个叶沿时间轴堆叠.
    all_trajs, in_trajs, out_trajs = {}, {}, {}
    # ``raw_trajs``: variable -> ndarray, 形状为 (S, ...), 保存模型未投影的轨迹.
    raw_trajs = {}
    try:
        # ``key``: str, 当前堆叠的轨迹变量名, 依次为 ``node``、``pos``、``halfedge``.
        for key in traj_dict.keys():
            # ``all_trajs[key]``: ndarray, 形状为 (S_all, ...), 把 all 来源 CPU Tensor 沿阶段轴堆叠.
            # ``d``: CPU Tensor, 当前 ``key`` 在 all 来源中的一个阶段项; 实体形状由 ``key`` 决定.
            all_trajs[key] = np.stack([d.numpy() for d in traj_dict[key]], axis=0)  # torch is slower here
            # ``in_trajs[key]``: ndarray, 形状为 (T_total, ...), 只含每步带噪输入.
            # ``d``: CPU Tensor, 当前 ``key`` 在 in 来源中的一个采样步项; 实体形状由 ``key`` 决定.
            in_trajs[key] = np.stack([d.numpy() for d in in_dict[key]], axis=0)
            # ``out_trajs[key]``: ndarray, 形状为 (T_total, ...), 只含每步投影后状态.
            # ``d``: CPU Tensor, 当前 ``key`` 在 out 来源中的一个采样步项; 实体形状由 ``key`` 决定.
            out_trajs[key] = np.stack([d.numpy() for d in out_dict[key]], axis=0)
            # ``raw_trajs[key]``: ndarray, 形状为 (T_total, ...), 只含模型未投影预测.
            # ``d``: CPU Tensor, 当前 ``key`` 在 raw 来源中的一个采样步项; 实体形状由 ``key`` 决定.
            raw_trajs[key] = np.stack([d.numpy() for d in raw_dict[key]], axis=0)
    except RuntimeError:
        raise NotImplementedError('fix to save traj information')
        for key in traj_dict.keys():
            all_trajs[key] = pad_and_stack(traj_dict[key], dim=0)
            in_trajs[key] = pad_and_stack(in_dict[key], dim=0)
            out_trajs[key] = pad_and_stack(out_dict[key], dim=0)
            raw_trajs[key] = pad_and_stack(raw_dict[key], dim=0)
    
    # ``trajs``: dict[str, dict[str, ndarray]], 顶层来源为 ``all/in/out/raw``, 内层变量为 ``node/pos/halfedge``.
    trajs = {
        # ``trajs.all``: dict[str, ndarray], 保存 gt 与每步输入/输出交错组成的完整阶段轨迹.
        'all': all_trajs,
        # ``trajs.in``: dict[str, ndarray], 保存每步重新加噪后的输入轨迹.
        'in': in_trajs,
        # ``trajs.out``: dict[str, ndarray], 保存每步 M-Projector 投影后的状态轨迹.
        'out': out_trajs,
        # ``trajs.raw``: dict[str, ndarray], 保存每步模型未经投影的原始预测轨迹.
        'raw': raw_trajs,
    }
    # ``outputs.confidence_pos_traj``: FloatTensor, 形状为 (N, T_total); ``dim=-1`` 将采样步放在最后一维.
    outputs['confidence_pos_traj'] = torch.stack(cfd_traj, dim=-1)
    return batch, outputs, trajs

def pad_and_stack(tensor_list, dim):
    size_list = [tensor.size(0) for tensor in tensor_list]
    max_size = max(size_list)
    if tensor_list[0].dim() == 1:
        tensor_list = [F.pad(tensor, (0, max_size-size), mode='constant', value=-1)  # type = -1 means
                    for tensor, size in zip(tensor_list, size_list)]
    else:
        tensor_list = [F.pad(tensor, (0, 0, 0, max_size-size), mode='constant', value=0)
                    for tensor, size in zip(tensor_list, size_list)]
    return torch.stack(tensor_list, dim=dim)

def seperate_outputs2(batch, outputs, trajs, off_tqdm=False):
    """按 PyG 图归属把批级最终状态、模型输出和轨迹拆回单分子。

    输入参数:
        - batch: PyG Batch，最终采样状态，含 B 个候选图。
        - batch.num_graphs: int，批内真实图数 B。
        - batch.node_type: LongTensor，形状为 (N,)，批级最终原子类别。
        - batch.node_pos: FloatTensor，形状为 (N, 3)，批级最终坐标，单位 Å。
        - batch.halfedge_type: LongTensor，形状为 (H,)，批级最终完整半边类别。
        - batch.halfedge_index: LongTensor，形状为 (2, H)，批级 0-based 半边端点索引。
        - batch.node_type_batch: LongTensor，形状为 (N,)，每个原子的图归属索引。
        - batch.halfedge_type_batch: LongTensor，形状为 (H,)，每条半边的图归属索引。
        - batch.pocket_center: FloatTensor，按图存储的口袋中心，单图拆分后通常形状为 (1, 3)，单位 Å。
        - outputs: Mapping[str, Tensor]，最后一步模型输出；第 0 维长度 N/H/``len(batch)`` 决定现有启发式拆分分支。
        - trajs: Mapping|None，``sample_loop3`` 返回的 ``source -> variable -> Tensor|ndarray``；实体轴位于第 1 维。
        - trajs.<source>.node: Tensor|ndarray，形状为 (S, N)，批级原子类别轨迹。
        - trajs.<source>.pos: Tensor|ndarray，形状为 (S, N, 3)，批级坐标轨迹，单位 Å。
        - trajs.<source>.halfedge: Tensor|ndarray，形状为 (S, H)，批级半边类别轨迹。
        - off_tqdm: bool，是否关闭拆分进度条。

    返回值:
        - generated_list: list[dict]，长度为 B，每项是 ``FeaturizeMol.decode_output`` 的单图输入。
        - generated_list[b].node: ndarray，形状为 (N_b,)，图 b 的最终原子类别。
        - generated_list[b].pos: ndarray，形状为 (N_b, 3)，图 b 的最终局部坐标，单位 Å。
        - generated_list[b].halfedge: ndarray，形状为 (H_b,)，图 b 的最终半边类别。
        - generated_list[b].halfedge_index: ndarray，形状为 (2, H_b)，图 b 的 0-based 半边端点索引。
        - generated_list[b].pocket_center: ndarray，形状为 (1, 3)，图 b 的口袋中心，单位 Å；缺失或为空时为零。
        - outputs_list: list[dict]，长度为 B，每项保存启发式判定可拆分的 CPU Tensor 输出叶。
        - outputs_list[b].pred_node: Tensor，可选，形状为 (N_b, C_n)，图 b 的原子类别 logits。
        - outputs_list[b].pred_pos: Tensor，可选，形状为 (N_b, 3)，图 b 的坐标预测，单位 Å。
        - outputs_list[b].pred_halfedge: Tensor，可选，形状为 (H_b, C_e)，图 b 的半边类别 logits。
        - outputs_list[b].confidence_node: Tensor，可选，形状为 (N_b, 1)，图 b 的原子 confidence 原始输出。
        - outputs_list[b].confidence_pos: Tensor，可选，形状为 (N_b, 1)，图 b 的坐标 confidence 原始输出。
        - outputs_list[b].confidence_halfedge: Tensor，可选，形状为 (H_b, 1)，图 b 的半边 confidence 原始输出。
        - outputs_list[b].confidence_pos_traj: Tensor，可选，形状为 (N_b, T_total)，图 b 的坐标 confidence 轨迹。
        - outputs_list[b].halfedge_index: Tensor，形状为 (2, H_b)，图 b 的半边端点索引。
        - outputs_list[b].pocket_center: Tensor，通常形状为 (1, 3)，图 b 的口袋中心，单位 Å。
        - traj_list_dict: dict[str, list[dict]]|list，存在轨迹时顶层键为来源、每个列表长度为 B；``trajs is None`` 时为空列表。
        - traj_list_dict[source][b].node: ndarray，形状为 (S, N_b)，单图原子类别轨迹。
        - traj_list_dict[source][b].pos: ndarray，形状为 (S, N_b, 3)，单图坐标轨迹，单位 Å。
        - traj_list_dict[source][b].halfedge: ndarray，形状为 (S, H_b)，单图半边类别轨迹。

    注意:
        - 函数名沿用源码拼写 ``seperate``。
        - 输出归属按第 0 维长度启发式判断；若 N、H 与 Batch 字段数偶然相同，先命中的节点分支决定拆分方式。
        - 第三分支使用 ``len(batch)``；常见 PyG ``Batch.__len__`` 返回字段数而非 ``num_graphs``，因此该分支疑似把字段数误作图数，本学习分支只记录不修复。
    """

    # ``num_graphs``：int B，批内单图数量。
    num_graphs = batch.num_graphs
    try:
        # ``data_list``：长度 B 的单图 Data；PyG 会撤销 ``__inc__`` 添加的批索引偏移。
        data_list = batch.to_data_list()
    except RuntimeError:
        del batch['node_p2'], batch['halfedge_p2'], batch['halfedge_p1p2']
        # ``data_list``：删除 AR 专用变长字段后重新拆分；标准构象/docking 通常不进入。
        data_list = batch.to_data_list()

    # ``generated_list``：decode_output 所需的单分子 NumPy 叶字段列表。
    generated_list = []
    # ``i_mol``：int，当前图编号，取值范围为 ``[0, num_graphs)``，索引 ``data_list`` 与返回列表。
    for i_mol in tqdm(range(num_graphs), desc='Seperating mols', total=num_graphs, disable=off_tqdm):
        # if 'pocket_center' in data_list[i_mol].keys:
        if 'pocket_center' in data_list[i_mol]:
            # ``pocket_center``：ndarray，形状为 (1, 3) Å；把局部配体坐标还原到原始口袋系时使用。
            pocket_center = data_list[i_mol]['pocket_center'].cpu().numpy()
            if len(pocket_center) == 0:
                # ``pocket_center``：空中心 -> 形状为 (1, 3) 的零中心，便于坐标广播。
                pocket_center = np.zeros([1, 3])
        else:
            # ``pocket_center``：ndarray，形状为 (1, 3) 的零向量；没有口袋字段时构象坐标不发生额外平移。
            pocket_center = np.zeros([1, 3])
        # ``generated_list[b]``：dict，当前图供 ``decode_output`` 使用的最终状态与坐标还原中心。
        generated_list.append({
            # ``generated_list[b].node``：ndarray，形状为 (N_b,)，图 b 的最终原子类别索引。
            'node': data_list[i_mol]['node_type'].cpu().numpy(),
            # ``generated_list[b].pos``：ndarray，形状为 (N_b, 3)，图 b 的最终局部配体坐标，单位 Å。
            'pos': data_list[i_mol]['node_pos'].cpu().numpy(),
            # ``generated_list[b].halfedge``：ndarray，形状为 (H_b,)，图 b 的最终完整半边类别索引。
            'halfedge': data_list[i_mol]['halfedge_type'].cpu().numpy(),
            # ``generated_list[b].halfedge_index``：int64 ndarray，形状为 (2, H_b)，端点索引图 b 的原子维。
            'halfedge_index': data_list[i_mol]['halfedge_index'].cpu().numpy(),
            # ``generated_list[b].pocket_center``：ndarray，形状为 (1, 3)，解码时加回局部配体坐标的平移中心，单位 Å。
            'pocket_center': pocket_center,
        })
    
    # ``node_sizes``：list[int] 长度 B；第 b 项为图 b 的原子数 N_b。
    node_sizes = torch.bincount(batch['node_type_batch']).tolist()
    # ``halfedge_sizes``：list[int] 长度 B；第 b 项为图 b 的完整半边数 H_b。
    halfedge_sizes = torch.bincount(batch['halfedge_type_batch']).tolist()
    # ``outputs_dict``：输出叶名到长度 B 的 Tensor 元组或原图级张量的映射。
    outputs_dict = {}
    # ``key``：str，当前模型输出叶名。
    # ``value``：Tensor，当前输出叶；第 0 维可能沿节点、半边或图实体对齐。
    for key, value in outputs.items():
        if len(value) == len(batch['node_type_batch']):
            # ``outputs_dict``：节点级叶沿第 0 维按 N_b 拆成 B 份；保留其余通道/时间维。
            outputs_dict[key] = torch.split(value, node_sizes)
        elif len(value) == len(batch['halfedge_type_batch']):
            # ``outputs_dict``：半边级叶沿第 0 维按 H_b 拆成 B 份。
            outputs_dict[key] = torch.split(value, halfedge_sizes)
        elif len(value) == len(batch):
            # ``outputs_dict``：历史分支：第 0 维等于 Batch 字段数时原样保留；这不可靠地等同于图级 B。
            outputs_dict[key] = value

    # ``outputs_list``：长度 B 的单分子输出字典列表。
    outputs_list = []
    # ``i_mol``：int，当前图编号，用于从每个已拆分输出叶取第 b 份。
    for i_mol in range(num_graphs):
        # ``output``：单图 CPU Tensor 叶映射；所有被保留的叶都以 i_mol 取一份。
        # ``key``：str，当前已拆分模型输出叶名。
        # ``value``：tuple[Tensor]|Tensor，按图可索引的当前批级输出叶。
        output = {key:value[i_mol].cpu() for key, value in outputs_dict.items()}
        # ``output.halfedge_index``：LongTensor，形状为 (2, H_b)，图 b 的 0-based 半边端点索引。
        output.update({'halfedge_index': data_list[i_mol]['halfedge_index'].cpu()})
        # ``output.pocket_center``：FloatTensor，通常形状为 (1, 3)，图 b 的坐标平移中心，单位 Å。
        output.update({'pocket_center': data_list[i_mol]['pocket_center'].cpu()})
        outputs_list.append(output)
    
    if trajs is not None:
        # ``node_split_size``：list[int]，用于沿轨迹实体轴拆分 N_b。
        node_split_size = batch['node_type_batch'].bincount().tolist()
        # ``halfedge_split_size``：list[int]，用于沿轨迹实体轴拆分 H_b。
        halfedge_split_size = batch['halfedge_type_batch'].bincount().tolist()
        # ``key``：str，当前初始化的轨迹来源名，与 ``trajs`` 的一个顶层键相同。
        # ``traj_list_dict``：dict[str, list[dict]]，每个来源叶稍后填入 B 个单分子轨迹字典。
        traj_list_dict = {key:[] for key in trajs.keys()}
        # ``traj_who``：str，当前拆分的轨迹来源，例如 ``all/in/out/raw``。
        for traj_who in trajs.keys():
            if isinstance(trajs[traj_who]['node'], torch.Tensor):
                # ``node_split``：tuple[Tensor]，长度为 B，第 b 项形状为 (S, N_b)；轨迹实体轴固定为 ``dim=1``。
                node_split = torch.split(trajs[traj_who]['node'], node_split_size, dim=1)
                # ``pos_split``：tuple[Tensor]，长度为 B，第 b 项形状为 (S, N_b, 3)，坐标单位 Å。
                pos_split = torch.split(trajs[traj_who]['pos'], node_split_size, dim=1)
                # ``halfedge_split``：tuple[Tensor]，长度为 B，第 b 项形状为 (S, H_b)。
                halfedge_split = torch.split(trajs[traj_who]['halfedge'], halfedge_split_size, dim=1)
                # ``i_mol``：int，当前列表推导式中的图编号，索引三个按图拆分的 Tensor 元组。
                # ``traj_list_dict[traj_who]``：list[dict]，长度为 B；下列三个叶分别沿原子或半边实体轴拆分。
                traj_list_dict[traj_who] = [{
                    # ``traj_list_dict[traj_who][b].node``：ndarray，形状为 (S, N_b)，图 b 的原子类别轨迹。
                    'node': node_split[i_mol].cpu().numpy(),
                    # ``traj_list_dict[traj_who][b].pos``：ndarray，形状为 (S, N_b, 3)，图 b 的配体坐标轨迹，单位 Å。
                    'pos': pos_split[i_mol].cpu().numpy(),
                    # ``traj_list_dict[traj_who][b].halfedge``：ndarray，形状为 (S, H_b)，图 b 的完整半边类别轨迹。
                    'halfedge': halfedge_split[i_mol].cpu().numpy(),
                } for i_mol in range(num_graphs)]
            else:
                # ``indices_node``：ndarray，形状为 (B-1,)，节点累积切分点；不包含总长度终点。
                indices_node = np.cumsum(node_split_size)[:-1]
                # ``indices_halfedge``：ndarray，形状为 (B-1,)，半边累积切分点。
                indices_halfedge = np.cumsum(halfedge_split_size)[:-1]
                # ``node_split``：B 个 ndarray，形状为 (S, N_b)。
                node_split = np.split(trajs[traj_who]['node'], indices_node, axis=1)
                # ``pos_split``：B 个 ndarray，形状为 (S, N_b, 3) Å。
                pos_split = np.split(trajs[traj_who]['pos'], indices_node, axis=1)
                # ``halfedge_split``：B 个 ndarray，形状为 (S, H_b)。
                halfedge_split = np.split(trajs[traj_who]['halfedge'], indices_halfedge, axis=1)
                # ``i_mol``：int，当前列表推导式中的图编号，索引三个按图拆分的 ndarray 列表。
                # ``traj_list_dict[traj_who]``：list[dict]，长度为 B；NumPy 后端沿实体轴切出当前来源的单图轨迹。
                traj_list_dict[traj_who] = [{
                    # ``traj_list_dict[traj_who][b].node``：ndarray，形状为 (S, N_b)，图 b 的原子类别轨迹。
                    'node': node_split[i_mol],
                    # ``traj_list_dict[traj_who][b].pos``：ndarray，形状为 (S, N_b, 3)，图 b 的配体坐标轨迹，单位 Å。
                    'pos': pos_split[i_mol],
                    # ``traj_list_dict[traj_who][b].halfedge``：ndarray，形状为 (S, H_b)，图 b 的完整半边类别轨迹。
                    'halfedge': halfedge_split[i_mol],
                } for i_mol in range(num_graphs)]
    else:
        # ``traj_list_dict``：空 list，表示调用方未提供任何轨迹；该分支不返回来源键。
        traj_list_dict = []
    
    return generated_list, outputs_list, traj_list_dict


def get_cfd_traj(cfd_pos_traj, atom_dim=0, steps=100):
    """把逐原子逐步坐标置信度聚合为候选级轨迹分数。

    输入参数:
        - cfd_pos_traj: Tensor，通常形状为 (N_b, T_total)，叶值是 ``confidence_pos`` 的原始单通道输出，本函数不做 sigmoid。
        - atom_dim: int，原子轴；单分子默认第 0 维。
        - steps: int，一轮采样的步数 T；总轨迹能被 T 整除且不等于 T 时按轮重排。

    返回值:
        - cfd: float，标准路径先在原子轴求均值，再对选中轨迹的后一半时间步求均值。

    注意:
        - 多轮 refine 路径把“标准差大于 0.01 的轮数减一”当作轮索引；只有满足条件的轮恰好构成前缀时，它才等于最后一个满足轮的真实索引。
        - 没有任何轮满足标准差阈值时 ``last_round=-1``，因此选择末轮，而不是报告无有效轮。
    """

    # ``cfd_atoms``：Tensor，形状为 (T_total,)，逐步对 N_b 个原子求均值后的原始置信度。
    cfd_atoms = cfd_pos_traj.mean(dim=atom_dim)  # atom-wise mean
    if (len(cfd_atoms) != steps) and (len(cfd_atoms) % steps == 0):  # refine-based sampling
        # ``cfd_atoms_reshape``：Tensor，形状为 (R, T)，R 为采样/细化轮数。
        cfd_atoms_reshape = cfd_atoms.view(-1, steps)
        # ``std_cfd``：Tensor，形状为 (R,)，每轮沿 T 步的置信度标准差。
        std_cfd = cfd_atoms_reshape.std(dim=1)
        # ``last_round``：int，严格等于满足 std>0.01 的轮数减一，而非 argwhere 的末索引。
        last_round = len(std_cfd[std_cfd > 0.01]) - 1
        # ``cfd_atoms``：Tensor，形状为 (T,)，选中轮次的逐步候选级置信度。
        cfd_atoms = cfd_atoms_reshape[last_round]
    # ``cfd``：标量 Tensor，仅平均选中轨迹的后半段，降低早期高噪声步骤影响。
    cfd = cfd_atoms[-cfd_atoms.size(0)//2:].mean()
    if isinstance(cfd, torch.Tensor):
        # ``cfd``：转为 Python float，便于写入 pandas/CSV。
        cfd = cfd.item()
    return cfd

def post_process_generated(generated_list, outputs_list, traj_list_dict):
    mol_info_list = []
    for i_mol in range(len(generated_list)):
        mol_info = featurizer.decode_output(**generated_list[i_mol]) 
        mol_info.update(data_list[i_mol])  # add data info
        
        # reconstruct mols
        try:
            rdmol = reconstruct_from_generated_with_edges(mol_info)
            smiles = Chem.MolToSmiles(rdmol)
            if '.' in smiles:
                tag = 'incomp'
                pool.incomp.append(mol_info)
                logger.warning('Incomplete molecule: %s' % smiles)
            else:
                tag = ''
                pool.succ.append(mol_info)
                logger.info('Success: %s' % smiles)
        except MolReconsError:
            pool.bad.append(mol_info)
            logger.warning('Reconstruction error encountered.')
            smiles = ''
            # rdmol = Chem.MolFromSmiles(smiles)
            tag = 'bad'
            # raise NotImplementedError('fix to save information anyway')
            rdmol = create_sdf_string(mol_info)
        
        mol_info['rdmol'] = rdmol
        mol_info['smiles'] = smiles
        mol_info['tag'] = tag
        mol_info['output'] = outputs_list[i_mol]
        
        # get traj
        p_save_traj = np.random.rand()  # save traj
        if p_save_traj <  save_traj_prob:
            mol_traj = {}
            for traj_who in traj_list_dict.keys():
                traj_this_mol = traj_list_dict[traj_who][i_mol]
                for t in range(len(traj_this_mol['node'])):
                    mol_this = featurizer.decode_output(
                            node=traj_this_mol['node'][t],
                            pos=traj_this_mol['pos'][t],
                            halfedge=traj_this_mol['halfedge'][t],
                            halfedge_index=generated_list[i_mol]['halfedge_index'],
                            pocket_center=generated_list[i_mol]['pocket_center'],
                        )
                    mol_this = create_sdf_string(mol_this)
                    mol_traj.setdefault(traj_who, []).append(mol_this)
                    
            mol_info['traj'] = mol_traj
        mol_info_list.append(mol_info)


def get_mol_parts_linking(batch, device=None):
    parts = []
    for i_part in [1, 2]:
        # if f'gt_node_type_p{i_part}' not in batch.keys:
        if f'gt_node_type_p{i_part}' not in batch:
            continue
        part = {
            'gt_node_type': batch[f'gt_node_type_p{i_part}'].clone(),
            'gt_node_pos': batch[f'gt_node_pos_p{i_part}'].clone(),
            'gt_halfedge_type': batch[f'gt_halfedge_type_p{i_part}'].clone(),
        }
        parts.append(part)
    return parts
        

def seperate_outputs_no_traj(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge):
    outputs_pred = outputs

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        ind_halfedge = (batch_halfedge == i_mol)
        assert ind_node.sum() * (ind_node.sum()-1) == ind_halfedge.sum() * 2
        new_pred_this = [outputs_pred[0][ind_node],  # node type
                         outputs_pred[1][ind_node],  # node pos
                         outputs_pred[2][ind_halfedge]]  # halfedge type
                        
        halfedge_index_this = halfedge_index[:, ind_halfedge]
        assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
        halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

        new_outputs.append({
            'node': new_pred_this[0],
            'pos': new_pred_this[1],
            'halfedge': new_pred_this[2],
            'halfedge_index': halfedge_index_this,
        })
    return new_outputs



def get_atom_and_bond(pred_node, pred_pos, pred_halfedge):
    """
    Get the atom and bond information from the prediction (latent space)
    pred_node: [n_nodes, n_node_types]
    pred_pos: [n_nodes, 3]
    pred_halfedge: [n_halfedges, n_edge_types]
    """
    # get atoms
    pred_atom = torch.softmax(pred_node, dim=-1)
    atom_prob, atom_type = torch.max(pred_atom, dim=-1)
    is_atom = (atom_types > 0)
    n_atoms = is_atom.sum().item()
    atom_types = atom_types[is_atom] - 1
    atom_prob = atom_prob_all[is_atom]
    atom_prob_dummy = atom_prob_all[~is_atom]
    atom_pos = pos_pred[is_atom]
    
    # get bonds for real atom 
    n_context = data.ligand_context_pos.size(0)
    n_compose = data.compose_pos.size(0)
    bond_index = data.bond_index
    bond_mask_tbp = data.bond_mask
    # pred_bond_prob = torch.softmax(pred_bond_logits, dim=-1)
    pred_bond_prob = pred_bond_logits
    bond_prob_tbp, bond_types_tbp = torch.max(pred_bond_prob, dim=-1)
    index_real_in_compose = torch.nonzero(is_atom) + n_compose
    # n_atoms_real = n_compose + n_atoms 
    # bond_mask_real = torch.stack([((bond[0]<n_atoms_real) and (bond[1]<n_atoms_real)) for bond in bond_index.T])
    bond_mask_real = torch.from_numpy(np.array([(
        ((bond[0] in index_real_in_compose) or (bond[0] < n_context)) and
        ((bond[1] in index_real_in_compose) or (bond[1] < n_context))
    ) for bond in bond_index.T]))
    
    bond_index_real = bond_index[:, (bond_mask_tbp & bond_mask_real)]
    bond_mask = bond_mask_real[bond_mask_tbp]
    bond_types_real = bond_types_tbp[bond_mask]
    bond_prob_real = bond_prob_tbp[bond_mask]

    is_positive_bond = bond_types_real > 0
    bond_types = bond_types_real[is_positive_bond]
    bond_prob = bond_prob_real[is_positive_bond]
    bond_index = bond_index_real[:, is_positive_bond]
    bond_prob_dummy = bond_prob_real[~is_positive_bond]

    n_protein = data.protein_pos.size(0)
    idx_changer = torch.zeros(len(is_atom), dtype=torch.long)
    idx_changer[is_atom] = torch.arange(n_atoms) + n_context
    idx_changer = torch.cat([
        torch.arange(n_context),
        torch.zeros(n_compose - n_context),
        idx_changer
    ], dim=-1)
    bond_index = idx_changer[bond_index]
    # bond_index = torch.where(bond_index<n_context, bond_index,
    #                         (bond_index-n_protein)[idx_changer])
    return {
        'atom': [atom_types, atom_prob, atom_prob_dummy, atom_pos],
        'bond': [bond_types, bond_prob, bond_prob_dummy, bond_index]
    }

    # ele_types = frag_atom_max[is_atom] - 1  # minus 1 for 0 = None atom
    # # ele_prob = frag_atom_prob[is_atom]
    # pos = frag_pos[is_atom]


def add_ligand_atom_to_data(data, element, pos, bond_types, bond_index, type_map=[6,7,8,9,15,16,17]):
    """
    """
    data = data.clone()
    n_atoms = len(element)

    data.ligand_context_pos = torch.cat([
        data.ligand_context_pos,
        pos.view(n_atoms, 3).to(data.ligand_context_pos)
    ], dim=0)

    data.ligand_context_feature_full = torch.cat([
        data.ligand_context_feature_full,
        F.one_hot(element, len(type_map)).to(data.ligand_context_feature_full), # (n_atoms, num_elements)
    ], dim=0)

    element = torch.LongTensor([type_map[e] for e in element])
    data.ligand_context_element = torch.cat([
        data.ligand_context_element,
        element.view(n_atoms).to(data.ligand_context_element)
    ])

    data.ligand_context_bond_index = torch.cat([
        data.ligand_context_bond_index,
        bond_index.to(data.ligand_context_bond_index),
        torch.stack([
            bond_index[1], bond_index[0]
        ], dim=0).to(data.ligand_context_bond_index)
    ], dim=-1)
    data.ligand_context_bond_type = torch.cat([
        data.ligand_context_bond_type,
        bond_types.to(data.ligand_context_bond_type),
        bond_types.to(data.ligand_context_bond_type)
    ], dim=-1)

    return data


def get_atom_and_bond(data, pred_atom_logits, pred_bond_logits, pos_pred):
    # get atoms
    # pred_atom_prob = torch.softmax(pred_atom_logits, dim=-1)  # has NOT passed softmax
    pred_atom_prob = pred_atom_logits  # has passed softmax
    atom_prob_all, atom_types = torch.max(pred_atom_prob, dim=-1)
    is_atom = (atom_types > 0)
    n_atoms = is_atom.sum().item()
    atom_types = atom_types[is_atom] - 1
    atom_prob = atom_prob_all[is_atom]
    atom_prob_dummy = atom_prob_all[~is_atom]
    atom_pos = pos_pred[is_atom]
    
    # get bonds for real atom 
    n_context = data.ligand_context_pos.size(0)
    n_compose = data.compose_pos.size(0)
    bond_index = data.bond_index
    bond_mask_tbp = data.bond_mask
    # pred_bond_prob = torch.softmax(pred_bond_logits, dim=-1)
    pred_bond_prob = pred_bond_logits
    bond_prob_tbp, bond_types_tbp = torch.max(pred_bond_prob, dim=-1)
    index_real_in_compose = torch.nonzero(is_atom) + n_compose
    # n_atoms_real = n_compose + n_atoms 
    # bond_mask_real = torch.stack([((bond[0]<n_atoms_real) and (bond[1]<n_atoms_real)) for bond in bond_index.T])
    bond_mask_real = torch.from_numpy(np.array([(
        ((bond[0] in index_real_in_compose) or (bond[0] < n_context)) and
        ((bond[1] in index_real_in_compose) or (bond[1] < n_context))
    ) for bond in bond_index.T]))
    
    bond_index_real = bond_index[:, (bond_mask_tbp & bond_mask_real)]
    bond_mask = bond_mask_real[bond_mask_tbp]
    bond_types_real = bond_types_tbp[bond_mask]
    bond_prob_real = bond_prob_tbp[bond_mask]

    is_positive_bond = bond_types_real > 0
    bond_types = bond_types_real[is_positive_bond]
    bond_prob = bond_prob_real[is_positive_bond]
    bond_index = bond_index_real[:, is_positive_bond]
    bond_prob_dummy = bond_prob_real[~is_positive_bond]

    n_protein = data.protein_pos.size(0)
    idx_changer = torch.zeros(len(is_atom), dtype=torch.long)
    idx_changer[is_atom] = torch.arange(n_atoms) + n_context
    idx_changer = torch.cat([
        torch.arange(n_context),
        torch.zeros(n_compose - n_context),
        idx_changer
    ], dim=-1)
    bond_index = idx_changer[bond_index]
    # bond_index = torch.where(bond_index<n_context, bond_index,
    #                         (bond_index-n_protein)[idx_changer])
    return {
        'atom': [atom_types, atom_prob, atom_prob_dummy, atom_pos],
        'bond': [bond_types, bond_prob, bond_prob_dummy, bond_index]
    }

    # ele_types = frag_atom_max[is_atom] - 1  # minus 1 for 0 = None atom
    # # ele_prob = frag_atom_prob[is_atom]
    # pos = frag_pos[is_atom]


def get_next_step(
        data_list,
        generated,
        # transform,
        type_map=[6,7,8,9,15,16,17],
        threshold=None,
    ):
    new_data_list = []
    n_mols = len(data_list)
    # is_finished = np.zeros(n_mols, dtype=bool)
    start_bond = 0
    for i in range(n_mols):
        parent_sample = deepcopy(data_list[i])
        
        # # has focal
        # has_focal = generated['has_focal'][i]
        # if not has_focal:
        #     is_finished[i] = True
        #     new_data_list.append(parent_sample)

        #     n_bond = parent_sample.bond_mask.int().sum()
        #     start_bond = start_bond + n_bond
        #     continue
        # else:  # get focal informaiton
        #     focal_prob = generated['focal_prob'][i].item()

        # # get things to add ( pos, atom, bond)
        frag_pos = generated['pred_pos'][i]  # (max_fragment, 3)
        frag_atom_pred = generated['pred_atom'][i]  # (max_fragment, num_elements)
        n_bond = parent_sample.bond_mask.int().sum()
        bond_types = generated['pred_bond'][start_bond:start_bond+n_bond]
        start_bond = start_bond + n_bond
        #TODO: add pos traj information
        
        # # get real atoms
        # frag_atom_prob = torch.softmax(frag_atom_pred, dim=-1)  # has not passed softmax
        # frag_atom_prob, frag_atom_max = torch.max(frag_atom_prob, dim=-1)
        # is_atom = (frag_atom_max > 0)
        # n_atoms = is_atom.sum().item()
        # ele_types = frag_atom_max[is_atom] - 1  # minus 1 for 0 = None atom
        # # ele_prob = frag_atom_prob[is_atom]
        # pos = frag_pos[is_atom]
        things = get_atom_and_bond(parent_sample, pred_atom_logits=frag_atom_pred,
                                  pred_bond_logits=bond_types, pos_pred=frag_pos)
        atom_types, atom_prob, atom_prob_dummy, atom_pos = things['atom']
        bond_types, bond_prob, bond_prob_dummy, bond_index = things['bond']
        
        # # add to ligand
        data_new = add_ligand_atom_to_data(
            parent_sample,
            element = atom_types,
            pos = atom_pos,
            bond_types = bond_types,
            bond_index = bond_index,
            type_map = type_map
        )

        # # log
        if hasattr(data_new, 'prob_atom'):
            data_new.prob_atom.append(atom_prob.cpu().detach().numpy())
            data_new.prob_atom.append(atom_prob_dummy.cpu().detach().numpy())
        else:
            data_new.prob_atom = [atom_prob.cpu().detach().numpy()]
            data_new.prob_atom.append(atom_prob_dummy.cpu().detach().numpy())
        if hasattr(data_new, 'prob_bond'):
            data_new.prob_bond.append(bond_prob.cpu().detach().numpy())
            data_new.prob_bond.append(bond_prob_dummy.cpu().detach().numpy())
        else:
            data_new.prob_bond = [bond_prob.cpu().detach().numpy()]
            data_new.prob_bond.append(bond_prob_dummy.cpu().detach().numpy())

        new_data_list.append(data_new)

    return new_data_list


def add_bond_to_data(
        data,
        bond_index,
        bond_type
):
    bond_index_all = torch.cat([bond_index, torch.stack([bond_index[1, :], bond_index[0, :]], dim=0)], dim=1)
    bond_type_all = torch.cat([bond_type, bond_type], dim=0)
    data.ligand_context_bond_index = bond_index_all
    data.ligand_context_bond_type = bond_type_all
    return data

def finish_step(data, edge_pred, edge_index):
    edge_pred = torch.softmax(edge_pred, dim=-1)
    edge_type = edge_pred.argmax(dim=-1)
    edge_prob = edge_pred[torch.arange(len(edge_type)), edge_type]
    # drop edge_type == 0
    is_bond = (edge_type > 0)
    bond_index = edge_index[:, is_bond]
    bond_type = edge_type[is_bond]
    bond_prob = edge_prob[is_bond]
    # add to data
    data = add_bond_to_data(data, bond_index, bond_type)
    data.prob_bond = bond_prob
    return data
