# 正确中心T0-RA的完整推理与评价

本记录承接 [中心T0修复与重训](02-T0中心契约修复与重训.md)。正确训练的best为21600步，原C0 val/loss=1.7970343828201294；训练在31200步第三次学习率下降后正常完成。当前按用户2026-09-11更新的执行顺序，先完成本模型预先规定的全部推理、评价及结果记录，之后才启动下一个模型。B-C-T1-RA目前只登记了命令，没有提交运行。

## 已冻结的输入与预算

两份正式配置为 `configs/docking/sample-B-C-T0-RA-validation.yml` 与 `configs/docking/sample-B-C-T0-RA-test.yml`。两者读取本次正确训练保存的 `train_config/B-C-T0-RA.yml` 和 `checkpoints/step=21600.ckpt`，完整路径均位于 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/`，不读取旧错误运行。

模型身份为B-C-T0-RA，RA联合受体编码、center_translation=false；validation和test分别执行C0与C5。C0使用真实中心，C5使用清单中已有冻结偏移后的中心；原点和口袋在该条轨迹中固定，T0+C5保留实际给定中心并执行原高斯采样。首步纯先验，最终只加回模型原点一次；原self-ranking沿用原置信度、碰撞和立体化学项，不使用GT排序。

每实例每协议50候选、100步，初始采样batch_size=25。validation包含781个实例、两协议共78100条候选轨迹；test包含446个实例、两协议共44600条轨迹。合计122700条；测试ALL／CAP10／HF10_TO5为446／272／227个实例，三个视图复用同一候选池，验证只汇总ALL。失败候选不补生成、实例不从分母删除，纯核酸实例遵守同样口径。

## 正式运行命令

采样在已授权的371591、单张A800中顺序执行；使用项目冻结release，保留after_lock：

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T0-RA-validation
bash 训练与运行/sh/sample_docking.sh B-C-T0-RA-test
```

每个划分的两个协议全部采样结束后，从服务器项目根提交对应8核CPU评价任务。验证CPU评价可与测试GPU采样并行；不提前读取尚未完成的候选池生成最终汇总。

```bash
bash 训练与运行/submit_task.sh --sh evaluate_docking.sh --resource cpu --cpus 8 -- B-C-T0-RA-validation
bash 训练与运行/submit_task.sh --sh evaluate_docking.sh --resource cpu --cpus 8 -- B-C-T0-RA-test
```

以上均为正式命令。已完成的实现验收和短GPU测试详见日志00、02，不作为本次候选结果。当前仅新增实际best对应的两份配置，不修改采样、评价代码或共同清单；正式启动前核对配置解析值、实际输入路径、输出隔离和预算，并独立窄核本次配置。

主代理已完成本次两遍自查：没有新增或修改Python函数，第一遍核对配置职责、实际读取入口和模型来源，第二遍核对字段类型、预算、C0训练与C0／C5推理的区别及结果目录说明。本地使用原 `utils.misc.make_config` 成功加载两份YAML，与官方采样配置逐项比较冻结资产根、batch_size、候选数、步数、设备和CPU进程数；两份新配置除split及W&B名字外完全一致。该解析检查通过stdin交给 `tmp/pxm-20260910/venv/Scripts/python.exe -B -`，只加载配置，没有采样或评价。

服务器只读核对确认保存的训练配置为RA、center_translation=false、center口袋、72×1，明确best文件存在；推理输出根和T1训练目录均尚不存在。冻结清单重新只读计数仍为validation 781／118 PDB、test 446／77 PDB，测试子视图272／227。独立逻辑代理只窄核这两份正式配置、当前记录及必要消费入口，确认正确best、T0+C5、冻结种子、预算、输出隔离和最新逐模型顺序均符合契约，无阻塞项；没有重新进行第三轮全面代码审查。

## 产物和当前状态

独立输出根为 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-C0/`，validation／test、C0／C5分别建子目录。末尾C0标识正确C0训练来源，不限制推理条件。每实例保留poses.sdf、candidates.json、confidence.npz及result.json；评价增加candidate_metrics.json、assessment.json、各协议occurrences.json及summary.json。字段定义见 `docking/sampling.py` 和 `docking/evaluation.py` 的入口说明。

W&B评价使用 `pencounkdual-111/PocketXmol_raw`，validation和test独立run。采样阶段先保存本地逐实例产物；CPU评价完成后才创建对应W&B汇总。具体CPU作业号、完整运行耗时、结果与核酸占比分层在实际完成后填写。

正式配置及最新执行顺序保存于d788edb，经原sync_code.ps1非删除同步后，已在371591启动validation采样。启动请求记录为master时间2026-09-11 02:05:52；实际gnode09 launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RA_validation_job371591_20260911T020258`，release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_b3b44f062a12/PocketXMol`。master与gnode09时钟约有3分钟差异，两者原始时间分别保留，耗时使用程序单调计时，不跨节点时间相减。

启动元数据为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_validation_start.json`，实际动态命令留于同目录sample_B-C-T0-RA_validation_run_cmd.sh；本次out／err起始字节为56944267／73179。原控制器第12次执行，主进程19347；after_lock保留，try_lock已解除，kill_lock不存在。模型严格加载成功后已写入validation/run.json，冻结的科学配置明确RA、T0、C0／C5及21600步best。

首次确认前两个完整实例为5irx/0和5irx/1，均50／50成功、100步、每批25，实际分别完成200次批量forward；总耗时30.40／24.24秒，峰值张量显存1067199488／762195968字节。它们是本次正式验证结果，尚未进行RMSD或排名评价；不能由生成成功推断姿态正确。当前继续完整validation，test及CPU评价尚未启动，下一模型训练须等本文件结果收口。
