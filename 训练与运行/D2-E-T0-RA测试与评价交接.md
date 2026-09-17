# D2 包络训练后的正式测试与 CPU 评价交接

本文件为接手 agent 准备 D2-E-T0-RA 的后续命令，依据现行 [D2 实验日志](../日志/第二类实验（密度分支的训练）/3-D2.md) 和用户在侧对话中批准的交接准备。仅覆盖：等待当前训练正常结束、确认 best、完整测试集 E 推理、CPU 评价、更新原日志、资源回到等待。不是另一份实验日志，也不授权启动 local_cov、修改训练或接管其他作业。

**交接状态（2026-09-17）：用户已把D2包络剩余流程交给当前任务。** 现有训练保持不变，由独立只读监看代理检查状态；训练正常结束并返回`try_lock`后，主代理填写真实best并按本文执行E正式推理和CPU评价。当前动态命令只有训练，原冻结release没有D2包络测试配置；下面的占位配置不能原样启动。

## 交接与运行条件

用户已经明确移交后续操作责任。独立监看代理只使用简单命令读取进程、训练日志、checkpoint和锁状态，不修改文件、动态命令或作业。稳定训练时每60分钟检查一次，明确离完成较远时可以延长到90分钟；等待由多个300秒片段组成。训练继续使用原参数与在线W&B，不重启、不更换权重、不修改正在执行的动态命令。确认训练正常结束后只启动尚未完成的阶段。

| 核对项目 | 已核实的位置或身份 |
|---|---|
| 资源 | Slurm 显示 `379403_1`，实际 JobId `379403`，gnode10，单张 A800，16 CPU |
| GPU | 物理索引 1，UUID `GPU-f8f2f6b7-5bde-624a-fd2d-c0f4776730c9`；执行时以该 allocation 的 `CUDA_VISIBLE_DEVICES` 为准，不手工指定其他卡 |
| 控制目录 | `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379403` |
| 动态命令 | 控制目录内 `run_cmd_379403.sh` |
| 资源保留锁 | 控制目录内 `after_lock_379403`，始终保留 |
| 再次执行等待锁 | 控制目录的父目录内 `try_lock_379403` |
| 停止锁 | 控制目录内 `kill_lock_379403`；本次正常衔接不触碰 |
| 固定源码 | `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_e6173e5c817d/PocketXMol` |
| 源码来源 | 当前 D2 包络训练使用的固定 release，训练命令记录 Git `0c79d53336f16f103227b967e596562f8a272f3e` |
| 训练产物 | `/storage/penghongen/PocketXMol/training/D2-E-T0-RA` |
| 训练输出 | `/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/D2-E-T0-RA/train.out` 和 `train.err` |
| 训练 W&B | `pencounkdual-111/PocketXmol_density`，run id `psr0c3ok` |

训练正常完成的依据：`train.out` 出现最终 `Training result:`、`Training finished!` 和 `FORMAL_TRAIN_PROCESS_EXIT_ZERO`；最终停止原因与原规则一致；该训练进程退出，控制器返回 `try_lock`。异常退出不能视为训练完成。W&B 保持 online，无法确认在线记录正常时报告实际问题，不自行切 offline。

从最终 `Training result:` 读取 best 路径与最低 `val/loss`，并与完整 `checkpoints/last.ckpt` 中训练回调的 `best_model_path`、`best_model_score`、`pocketxmol.stop_reason` 对照。last 用于核实最终状态，**推理加载 best，不加载 last**。现有只读检查参考为 `tmp/formal-execution-20260914/read_completed_d2_center.sh`；如借用其检查方式，必须改为 D2-E 的训练目录，不能把中心模型的输出当作包络 best。无需重跑 GPU 预实验。

## 正式测试配置：结束训练后补齐唯一待定值

采样与评价直接使用固定 release 中已有的 Python 入口，它们接受外置 YAML 路径。这样不需要修改旧 release，也不会误用 local_cov 工作区。现有 `sample_docking.sh` 只接受 release 内的配置名称，因此本次使用其底层已有 Python 入口，不给这个 shell 脚本传外置路径，不新建采样框架。

最终配置保存为：

`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/D2-E-T0-RA-test/sample-D2-E-T0-RA-test.yml`

下列模板继承已经完成的 D1 包络测试配置字段，并将模型、训练配置与输出指向 D2 包络；与已完成的 D2 中心测试保持相同公共资产和科学预算。**只有 `checkpoint` 待填写**。接手时先核实此目录和正式候选目录是否已经存在；如其他 agent 已启动，不重复提交，也不覆盖配置或产物。

```yaml
# scripts/sample_docking.py 生成 E 候选；scripts/evaluate_docking.py 用同一配置评价。
model_name: D2-E-T0-RA
# str，读取该次训练保存的 D2 模型、原 T0 噪声及特征化设置，不另加密度覆盖参数。
train_config: /storage/penghongen/PocketXMol/training/D2-E-T0-RA/train_config/D2-E-T0-RA.yml
checkpoint: __训练正常结束后填写best_model_path__  # str，必须替换为最终真实 best 的完整路径。
receptor_branch: RA  # str，蛋白与核酸使用联合受体图。
dataset:
  root: /storage/penghongen/AdaLigand/Ori_Data
  derived_root: /storage/penghongen/Adaligand_Build/Ori_Data/pocketxmol
  smiles_root: /storage/penghongen/AdaLigand/Ori_Data/smiles_assets
  smiles_coords_root: /storage/penghongen/AdaLigand/Ori_Data/smiles_assets/SMILE_coords/v1
  manifest_root: /storage/penghongen/PocketXMol/data/smiles-v1/frozen
  knn: 32  # int，受体图的近邻上限；以上目录均沿用既有资产。
protocols: [E]  # list[str]，仅真实配体包络，不执行 C0/C5 或完整验证集采样。
split: test  # str，446 个冻结测试实例；三个测试视图共用候选。
output_root: /storage/penghongen/PocketXMol/sampling/D2-E-T0-RA
batch_size: 50  # int，同一实例的推理候选批量。
num_candidates: 50  # int，每实例候选预算，失败仍保留评价分母。
num_steps: 100  # int，原采样步骤数。
device: cuda  # str，使用本作业 GPU，沿官方 FP32 张量路径。
evaluation_workers: 8  # int，随后使用同一作业 CPU 的评价进程数。
wandb:
  entity: pencounkdual-111
  project: PocketXmol_density
  name: D2-E-T0-RA_test
  mode: online  # str，评价在线记录；凭据不写入配置或日志。
```

E 口袋仍按残基重原子质量中心到任一真实配体重原子距离严格小于 10 Å 选择；模型原点为实际输入受体重原子的均值。密度裁块允许按真实配体质心定位，裁块内缩不改变模型原点。RA、精确 SMILES、原 T0 采样、置信度及 self-ranking 均从固定实现继承。

## 正式运行命令

以下命令须由 `379403` 的 Slurm allocation 执行，不能在普通 SSH 会话中承载模型推理。接手 agent 在确认训练完成并返回等待后，按真实锁协议安装本次动态命令；仅在动态命令和配置就绪后消费父目录 `try_lock_379403`，保留 `after_lock_379403`，不使用 `scancel`。

动态命令的运行环境沿用现有入口，先设置：

```bash
set -euo pipefail
release_root=/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_e6173e5c817d/PocketXMol
run_root=/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/D2-E-T0-RA-test
pxm_python=/storage/penghongen/PocketXMol/runtime/venv/bin/python
sample_config="$run_root/sample-D2-E-T0-RA-test.yml"
cd "$release_root"
export PYTHONPATH="$release_root" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/storage/penghongen/tmp WANDB_MODE=online
```

**正式完整测试推理命令：**

```bash
"$pxm_python" scripts/sample_docking.py "$sample_config"
```

**推理正常退出后，同一配置的正式 CPU 评价命令：**

```bash
CUDA_VISIBLE_DEVICES='' "$pxm_python" scripts/evaluate_docking.py "$sample_config"
```

将两条正式命令按这个顺序串接在同一个动态命令中，保持 `set -e`，使推理非零退出时不继续评价。采样标准输出和错误分别留在 `run_root/sample.out`、`sample.err`，评价分别留在 `evaluate.out`、`evaluate.err`；分别记录起止时间和成功退出标记。第一次派发前将旧训练动态命令保存到 `run_root/previous_run_cmd.sh`，之后不覆盖它。恢复或再尝试时保留之前输出，不用新的 `>` 截断已有日志。

沿用已有 `训练与运行/runtime/create_launch.sh` 保存实际动态命令与 release/资源身份，采样和评价入口分别留证；`TASK_ARRAY_SPEC=1`、16 CPU、单 GPU、实际 JobId 379403。可参照 `tmp/formal-execution-20260914/start-d2-center-test-378693.sh` 的留证和输出组织，但不能原样运行该文件：它会操作 378693，并引用中心模型及固定旧 best。当前文档没有提前改写任何服务器控制文件。

## 只读核查与配置验收命令

本节命令不启动正式实验，与上一节分开记录。下面是 gnode10 上的轻量检查；Windows 可通过用户级 `Invoke-ProjectSsh.ps1 -TargetHostName gnode10 -Command ...` 调用，不打印凭据。

```bash
squeue -h -j 379403 -o '%i %T %N'
nvidia-smi -i GPU-f8f2f6b7-5bde-624a-fd2d-c0f4776730c9 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader
tail -c 2500 /storage/penghongen/tmp/pxm_formal_execution_20260914/runs/D2-E-T0-RA/train.out
tail -n 8 /storage/penghongen/tmp/pxm_formal_execution_20260914/runs/D2-E-T0-RA/train.err
```

检查实际进程、`run_cmd`、父目录 `try_lock` 和当前产物，确认没有别的执行者已经启动相同任务。配置正式落盘后，用原 Python 环境只读解析 YAML，确认占位符已经被最终 best 替换、路径可读，协议仅 E，batch=50、候选=50、步骤=100、CPU 进程=8、W&B=online，训练配置为 D2-E。核对固定 release 的入口和最终动态命令，后者用 `bash -n` 检查语法；这属于启动验收，不是新的科学实验。

## 完成标准与原日志回填

正式候选根为 `/storage/penghongen/PocketXMol/sampling/D2-E-T0-RA/test`。读取其中的 `run.json` 确认实际模型、best 和配置；E 的 446 个实例都必须有处理记录，50 候选预算及输入失败均保留，不以“必须全部成功生成”作为完成条件。推理进程退出成功之后执行 CPU 评价；核对 `summary.json` 的 E 协议、ALL/CAP10/HF10_TO5 三视图、失败分母及 W&B `online_completed` 状态。

每次取得新进度，写回 [原 D2 实验日志](../日志/第二类实验（密度分支的训练）/3-D2.md) 和 [密度总日志](../日志/第二类实验（密度分支的训练）/总日志.md)。完成时补齐训练停止原因、最终 best/损失、固定 release、外置配置路径、实际 launch、正式命令、测试与 CPU 评价时间、指标及 W&B；失败或恢复放在原 D2 日志后部，不新建逐尝试日志。根 [总日志](../日志/总日志.md) 只维护阶段状态，明确阶段完成时再写 handoff。

全部完成后确认控制器返回父目录 `try_lock_379403`，保留 after_lock、全部历史产物和资源，不追加任务。既有 D1/D2 阶段的记录与 Git 收口由接手 agent 按现行双线规则完成；不能把未完成的 local_cov 修改混入本阶段收口。本次侧对话只新增这份交接说明，不提交 Git，不修改共享实验日志或正在运行的任务。
