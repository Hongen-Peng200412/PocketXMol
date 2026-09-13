# F-5：中心T1、RB的独立训练与测试

本记录依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)及用户2026-09-11新增资源授权，执行既定六模型中的第5个模型B-C-T1-RB。任务由“核查PocketXMol执行前准备（3: 独立做另外半边）”负责；用户接受与另一对话重复计算，但各次产物必须独立。本文不改变共同数据、T1公式、RB构造或实验预算。

## 正式配置与命令

配置为 `configs/docking/F-5.yml`，科学与资源参数复制自B-C-T1-RB.yml，仅W&B显示名称改为F-5_B-C-T1-RB。使用官方初始参数、独立优化器和调度状态；训练动态C5、监督验证冻结C5，T1训练平移及后续采样中心修正均保留。名义global batch为72，单卡batch72、累积1、bf16；每800次优化器更新原val/loss，沿用已冻结停止规则。DataLoader暂保留已验收的15个worker，32核是本作业申请量。

在服务器 `/home/penghongen/My_Project/PocketXMol` 提交：

```bash
bash 训练与运行/submit_task.sh --sh train_docking.sh --resource h100 --gpus 1 --cpus 32 --after_hold --job-name F-5 -- F-5
```

该作业中的正式训练命令为：

```bash
bash 训练与运行/sh/train_docking.sh F-5
```

训练产物为 `/storage/penghongen/PocketXMol/training/F-5/`，首次配置位于train_config/F-5.yml；checkpoints保留定期检查点及last，W&B位于pencounkdual-111/PocketXmol_raw、名称F-5_B-C-T1-RB，由入口生成独立run id。不得使用另一对话的B-C-T1-RB目录续训或混入候选。

训练完成后读取本任务实际最低原val/loss的best，再建立明确的测试配置并登记命令；预定测试输出根 `/storage/penghongen/PocketXMol/sampling/F-5/`。仅测试C0/C5，每实例每协议50候选、100步、batch50；ALL/CAP10/HF10_TO5复用候选，分别为446/272/227个实例。随后完成8进程CPU评价和结果记录。不运行完整验证集采样或评价。

## 验收与执行状态

两遍主代理自查限定于新增配置、命令及输出隔离：第一遍用原make_config解析，确认除W&B名称外与既定模型配置相同；第二遍核对配置注释、训练入口的默认输出根、恢复限制及实验命名。没有新增或修改Python函数。Windows首次配置解析因默认GBK读取UTF-8注释失败；使用Python的-X utf8后通过，未修改项目解析器。

上述配置检查通过本地Python stdin执行，不加载模型、不读测试样本，不是正式训练或测试命令。原实现及T0修复的必要验收见日志00、02。独立代理对本次三份配置与三份日志完成两轮限定范围审查：配置等价、命令、科学协议和输出隔离通过；官方失败产物说明已修正，最新unstaged规则经窄核关闭，无剩余问题。

已于master时间2026-09-11 16:26:07提交，实际JobId为 **378587**，Slurm名称F-5，单张H100、32核、after_hold，无pre_hold。提交后首次squeue为PENDING，原因为Priority；实际release/launch及W&B id待启动后核实。提交请求、完整命令、返回码和作业编号保存在 `/storage/penghongen/PocketXMol/control/parallel-20260911/F-5_submit.json`。当前科学源码基准为f856ad8，本次F-5.yml及日志为未提交新增文件；安全同步成功，未删除远端文件。

H100仅申请这一张，使用after_hold；结束后保留after_lock。排队及稳定运行期间按用户要求用多次Start-Sleep -Seconds 300组成60或90分钟静默等待，醒来检查并报告有意义变化。用户随后明确：本对话全部新增及修改文件始终保持unstaged，不执行git add或git commit；不处理其他人的修改或暂存内容。运行来源以基准提交f856ad8、本次未提交文件及实际release/launch共同留证。

## 计划与实现差异

用户明确新增并行资源并允许重复计算；本任务使用独立配置名称、输出根及W&B记录，科学定义与既定第5模型相同。训练、测试、CPU评价及最终结果尚未完成。
