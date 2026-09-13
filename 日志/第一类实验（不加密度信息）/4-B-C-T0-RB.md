# 4-B-C-T0-RB

**正式训练已完成，C0／C5完整测试正在推理。** 截至15:01，C0已完成2／446实例、100个候选，全部成功；C5按既定顺序在C0之后执行，尚未开始。最终姿态指标须待完整推理及CPU评价。训练在31200次更新后因第三次学习率下降停止，实际best为21600步、原C0 val/loss=1.8065478801727295；它是全部39个定期检查点的最低原验证损失，1236个模型参数键均有限。全部检查点、last及训练W&B记录保留。

更新核查：2026-09-13 15:01（服务器 master，UTC+8）。依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md) 和 [边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)，本文件统一保存本模型的训练、测试、评价与尝试历史。共同准备见 [实现与共同数据准备](../实现与共同数据准备.md)，全实验进度见 [总日志](../总日志.md)。

## 当前有效运行与产物

| 项目 | 当前事实 |
|---|---|
| 训练产物根 | `/storage/penghongen/PocketXMol/training/B-C-T0-RB/` |
| 正式测试检查点 | `checkpoints/step=21600.ckpt`，相对于上述训练根；原C0 val/loss=1.8065478801727295 |
| 训练 W&B | [hthglbuy](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/hthglbuy) |
| 测试与评价产物 | `/storage/penghongen/PocketXMol/sampling/B-C-T0-RB/test/`；run.json及C0前两个实例的完整候选已产生，评价尚未开始 |
| 评价 W&B | 尚未创建 |

训练保留原路径 `val/loss` 和最低损失检查点选择；训练结束后直接完整测试，不做训练后完整验证集采样。每实例每协议50候选、100步，原 self-ranking（原置信度及碰撞、立体化学项组成的候选排序）不变。完整结果优先放在下文，执行核查和失败尝试放在后部。

ALL为446个实例的完整测试集合。按完整模板身份object_key在ALL中的频数分组，只有频数大于10的身份才沿冻结顺序截取：CAP10保留前10个，HF10_TO5保留前5个；频数不超过10的身份全部保留。两个视图分别272和227个实例，与ALL复用候选，不重复生成或相加计数。

## 最近一次进度与下一步

采样主进程39974属于job_371591，实际run.json与已审查的RB、T0、C0／C5及best21600配置一致。C0中的11jb/0、11jb/1各完成50个候选和100次批量forward，耗时34.03／33.38秒，峰值张量显存约2.54 GB；没有NaN、Traceback、OOM或降低batch记录。after_lock保留，try_lock不存在，训练产物未覆盖。

完整训练摘要为 `/storage/penghongen/PocketXMol/training/B-C-T0-RB/training_summary_20260913.json`。继续完成446实例的C0和C5测试，再用同作业8个CPU评价进程完成三个视图及核酸占比报告，不安排完整验证集采样。

## 正式测试配置与命令

`configs/docking/sample-B-C-T0-RB-test.yml`读取本次保存的训练配置及21600步best，receptor_branch=RB、center_translation=false、protocols=[C0,C5]、split=test。与正确T0-RA的测试配置相比，仅替换RB分支、模型名称及各自训练／产物身份；冻结C5偏移、种子、50候选、100步、batch50和8进程评价保持。

中心推理始终按实际给定中心选袋及定原点；T0+C5保留冻结偏移后的输入中心，首步及后续均使用原T0高斯链，不强制候选质心归零。定位后采样器不读取真值中心或偏移，输出只加回模型原点一次，不做最终质心对齐。

本次没有新增或修改生产Python函数。主代理第一遍核对配置读取关系、模型与数据分支、检查点和输出身份，第二遍核对YAML注释及日志含义；独立代理随后在同一范围完成两轮只读核查，均通过。YAML解析比较确认相对正确T0-RA仅有六个模型身份、路径与分支字段变化，科学条件及预算保持；未扩大既有生产代码审查。

以下正式采样命令已于2026-09-13接入371591执行；release、launch和启动记录见下文。

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T0-RB-test
```

两个协议的全部候选完成后，在同一A800作业执行以下正式CPU评价命令（尚未执行）：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T0-RB-test
```

## 中心T0与RB契约

配置为 `configs/docking/B-C-T0-RB.yml`。训练及原val/loss均使用C0：给定中心和模型原点C为完整配体重原子几何中心g，按标准受体残基重原子质量中心到g的距离严格小于15 Å选取完整残基。配体和受体共同减去C，局部监督目标x*的几何中心为0；只执行原高斯 `x_in=x*+s*σ(N)*ε`，s为1-level_dict['pos']，沿用原GaussianExplodePrior的尺度和逐原子噪声。center_translation=false，训练不抽新增中心偏移、不增加整分子平移。

RB分别构建和编码标准蛋白与核酸受体图，从官方权重初始化后复制蛋白编码器参数给核酸编码器。保留原主体模型、loss、置信度头及其目标、同构重分配和原固定字段恢复。

正式测试沿用446个实例的C0和冻结C5条件。T0+C5保持实际给定的偏移中心选袋及定原点，后续仍仅用T0原高斯；首步使用原纯高斯先验，不强制候选实际质心归零。定位后采样器不读取真值中心或偏移，输出只加回实际模型原点一次，不进行最终质心对齐。

## 固定训练配置与来源

源码使用 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_43742cdf8166/PocketXMol`，基线ec06dbd已含中心T0纠偏，追加c9cc4a1仅处理空E数据流，不改变中心模型。中心T0完整路径验收、原官方权重及72×1训练验收已经通过；RA／RB模型接线和既有生产代码已完成两遍自查及两轮独立审查。本次没有新增或修改生产函数，不扩大科学范围或重复验收。

配置解析核对：相对同release的B-C-T0-RA.yml，只将model.nucleic_branch改为RB、train.wandb.name改为B-C-T0-RB，其余字段一致；pocket_mode=center、center_translation=false，实际DataModule据此选C0训练与监督验证。冻结训练清单65290、验证781、校准361、测试446保持，E专用空口袋跳过规则不改变中心数据流。目标训练根已确认不存在，正式接入时再次核对。

训练从规定官方pocketxmol.ckpt参数开始，不传--resume，独立创建AdamW、调度器及W&B身份。batch72、累积1、bf16、15个数据worker，有效全局批量72。AdamW初始lr=1e-4、weight_decay=0.001、betas=0.99／0.999、eps=1e-8、warmup=0；每800次优化器更新计算原val/loss，Plateau相对阈值1%、patience5、factor0.2，第三次实际下降立即停止，上限40000步。best按最低原C0 val/loss选择，last和全部定期检查点保留。

## 正式训练命令与产物

以下是正式训练命令，产物核对或验收命令另行记录：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RB
```

独立训练根为 `/storage/penghongen/PocketXMol/training/B-C-T0-RB/`，保存的训练配置及检查点分别位于train_config和checkpoints子目录。W&B使用pencounkdual-111/PocketXmol_raw、名称B-C-T0-RB，实际run id为hthglbuy。

371591控制目录仍为 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`。前一E评价正常结束且try_lock恢复后，保存新旧动态命令与启动记录，再移除try_lock运行。after_lock和此前有效及异常产物全部保留，不使用scancel。

训练完成后直接用实际best测试C0／C5，每实例每协议50候选、100步、batch优先50；之后用同一作业8个CPU评价进程完成三个视图和核酸比例分析。明确best的采样配置及短正式命令在训练结束后登记，不执行训练后完整验证集采样，不额外增加实验。

## 正式接入

2026-09-13 master时间06:14:42保存新旧动态命令及启动记录，移除try_lock后由371591控制器第21次执行启动。接入前再次确认目标训练根不存在、E-RA评价进程退出且W&B汇总成功，现有动态命令确为本任务E-RA评价入口。after_lock保留，没有使用kill_lock。

实际源码release为43742cdf8166，登记提交3101cff；配置SHA256为19809af372ac4c127efd8dab8c3850992cbb1164211a258eaebf4a4b0d6f8279。launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T0-RB_job371591_20260913T061132`。启动记录 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T0-RB_start.json`，同目录保存train_B-C-T0-RB_run_cmd.sh和修改前的train_B-C-T0-RB_before_run_cmd.sh。累计out／err起点113849959／311189。

实际主进程7586和15个数据worker均属于371591，命令行没有--resume，训练输出指向独立B-C-T0-RB根。日志确认从规定官方pocketxmol.ckpt加载模型，使用bf16，全部模型参数参与训练。W&B新运行是[hthglbuy](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/hthglbuy)，本地记录为训练根wandb/run-20260913_061159-hthglbuy。启动检查28步时原损失及置信度损失均有限，没有NaN、Traceback、OOM或reduce_batch；完整训练和原C0验证继续执行。

## 计划与实现差异

本实验沿用已纠正的中心T0契约和既定RB配置，没有新增科学开关、代码或资产重建。完整训练及best核对已完成，C0／C5测试已接入，完整推理及报告仍待完成。

## 正式C0／C5测试接入

2026-09-13 master时间14:58:53保存新旧动态命令及启动记录，移除恢复的try_lock，由371591控制器第22次执行接续测试。接入前确认训练主进程7586退出、前一动态命令仍为本模型训练入口、独立采样根不存在；after_lock保留，未使用kill_lock，没有覆盖训练产物。

实际release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_07ea8b8a2406/PocketXMol`，以已审查43742cdf8166源码副本增加本次测试配置，生产代码不变。配置提交933a49c，LF字节SHA256为5968646c8d56be437a0131df0110ab22bce611bd4d242d6a6b66b55478499650；创建release时再次核对保存的训练配置、实际best及全部测试预算。

实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RB_test_job371591_20260913T145551`，节点时钟比master慢约3分钟。启动记录 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RB_test_start.json`，同目录保存sample_B-C-T0-RB_test_run_cmd.sh和修改前的sample_B-C-T0-RB_test_before_run_cmd.sh。当前采样out／err从149921972／321399字节读取，避免混入此前训练日志。采样主进程39974的Slurm cgroup已核对。

## 训练完成与检查点只读核对

当前训练日志从 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/out` 和 `err` 的113849959／311189字节起读取。全部39次原C0验证正常保存，31200步验证后正常停止，W&B hthglbuy上传完成。训练进度栏耗时8小时13分35秒，平均约1.05次更新／秒；未见Traceback、OOM、非有限损失或reduce_batch记录。

在371591.8的8核CPU内加载last和实际best，确认stop_reason=plateau、decline_count=3、global_step=last_validation_step=31200，优化器和调度器末次学习率均为8.000000000000002e-7，第三次下降后没有继续更新。best21600对应的1.8065478801727295是全部39个定期检查点的最低原验证损失；39个检查点均存在，best的1236个model参数键全部有限。

以下是已执行的产物核对命令，不是正式训练、推理或评价命令。脚本只读取已有检查点并新写training_summary_20260913.json，不改写检查点；本地部署记录为 `tmp/pxm-20260913/check_c_t0_rb_checkpoint.sh`，摘要副本为 `tmp/pxm-20260913/B-C-T0-RB-training-summary.json`。

```bash
srun --jobid=371591 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_checkpoint_20260913/inspect_c_t0_rb.py
```

## 实验过程与失败尝试

本模型尚无失败或中断的正式尝试；它直接使用已纠正的C0训练契约，不继承错误T0-RA参数。
