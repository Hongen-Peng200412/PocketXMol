# B-E-T0-RA-SMILES

本轮执行边界（2026-09-16）：仅收口D1、D2、修正编码的无密度RA＋T0中心／包络及官方冻结对照；完成部分不重跑。D3、D4本轮取消，禁止local_cov及额外实验。各资源完成自身范围内全部测试与CPU评价后，保留after_lock、资源及历史产物，回到try_lock等待。后部旧排程仅作历史。

当前状态（2026-09-16 08:52／08:54）：B-E-T0-RA-SMILES：全流程完成，best21600；E测试446实例、22300候选生成和评价成功，Top-1=63.00448%，评价W&B t5ckbgr9 online_completed。

| 项目 | 当前内容 |
|---|---|
| 资源 | 379402_0，实际JobId379402，gnode09，A800，16CPU |
| 配置 | configs/docking/B-E-T0-RA-SMILES.yml |
| 产物根 | /storage/penghongen/PocketXMol/training/B-E-T0-RA-SMILES |
| best检查点 | /storage/penghongen/PocketXMol/training/B-E-T0-RA-SMILES/checkpoints/step=21600.ckpt；val/loss=1.5245567560195923 |
| 完整恢复检查点 | /storage/penghongen/PocketXMol/training/B-E-T0-RA-SMILES/checkpoints/last.ckpt；31200更新，第三次下降停止 |
| 训练W&B | https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/2uv4mdnt；online finished |
| 测试配置与产物 | configs/docking/sample-B-E-T0-RA-SMILES-test.yml；/storage/penghongen/PocketXMol/sampling/B-E-T0-RA-SMILES/test |
| 测试源码 | 0f09526ea210b573071c1c601881c16412b7999a；/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_0beb73604a7d/PocketXMol |
| 测试协议 | E；每实例每协议50候选、100步，batch50 |
| best／W&B／指标 | B-E-T0-RA-SMILES：全流程完成，best21600；E测试446实例、22300候选生成和评价成功，Top-1=63.00448%，评价W&B t5ckbgr9 online_completed |

## 完整E测试结果

来源为`/storage/penghongen/PocketXMol/sampling/B-E-T0-RA-SMILES/test/summary.json`；E目录保留逐实例`occurrences.json`及候选、评价文件。本地只读副本为`tmp/formal-execution-20260914/B-E-T0-RA-SMILES-summary.json`。评价W&B为[包络复验测试](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/t5ckbgr9)，summary确认online_completed、error=null。

成功定义为RMSD<2 Å，Top-1／Top-5沿原self-ranking，Oracle为50候选中最佳RMSD。前三项实例等权，最后一列先在PDB内平均再在PDB间等权；三个视图复用同一候选池。

| 视图 | 实例 | Top-1 | Top-5 | Oracle | PDB等权Top-1 |
|---|---:|---:|---:|---:|---:|
| ALL | 446 | 63.00% | 79.15% | 89.69% | 63.98% |
| CAP10 | 272 | 56.99% | 72.43% | 86.40% | 62.40% |
| HF10_TO5 | 227 | 57.71% | 72.25% | 87.22% | 62.51% |

ALL共22300候选，生成和RMSD评价均成功，无失败实例，Top-1平均RMSD=2.78879 Å。CPU评价于2026-09-16 05:05:03开始、05:58:22正常退出，约53分钟，使用本资源8进程。各实例评价耗时之和是并行CPU时间，不能当作该墙钟时长。

所有新训练从规定官方参数初始化，重建优化器与调度状态；训练bf16-mixed，推理官方FP32张量路径。训练保留原val/loss与best选择，结束后直接完整测试，中心C0/C5、包络E；既有清单、视图、C5和种子不重建，每实例每协议50候选、100步，推理优先batch_size=50。W&B保持online，无法在线时报告，不自行切换offline。 无密度和D1起始72×1，D4/D2/D3起始36×2；OOM按72→36→24降批、累积相应1→2→3，配置global_batch_size保持72。接受原reduce_batch偶发裁批和累积梯度丢失，不增加补样或梯度补偿。仍无法运行则报告实际问题。

## 正式运行命令

以下命令已于16:58:48派发；17:02:10正式开始，W&B 2uv4mdnt online。

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RA-SMILES
```

以下为训练结束后已批准的完整E测试及CPU评价命令；使用best21600，执行状态见当前状态与后文记录。

```bash
bash 训练与运行/sh/sample_docking.sh B-E-T0-RA-SMILES-test
bash 训练与运行/sh/evaluate_docking.sh B-E-T0-RA-SMILES-test
```

评价只在采样正常退出后执行，沿用同一配置；不采样完整验证集。

## 只读核查与验收依据

前置验收已被接受，不重跑。启动前核对真实作业、GPU UUID、控制目录、after_lock及try_lock；原命令独立备份。

启动前核查（16:57）：379402仅控制器等待，实际GPU UUID GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b，显存5 MiB、利用率0%；after_lock及父目录try_lock存在，kill_lock不存在。本包络训练目录尚不存在，已验收配置为E、72×1、global72、bf16-mixed、W&B online及规定官方初始权重；不传resume，不复用中心或旧包络状态。正式执行沿用已验收release e6173e5c817d（来源0c79d53），保留原控制命令及全部前序产物。

## 本次正式启动记录

正式测试确认（2026-09-16 01:19）：stdout记录实际开始01:17:57、job379402、上述release及launch；首实例11jb/0完成50候选，GPU持续计算，stderr无报错。控制器等待到执行的原仓库快照过程未触发重启，没有覆盖旧模型或产物。

E测试派发（2026-09-16 01:16:20，gnode09）：使用冻结release 0beb73604a7d，Git archive SHA256为c8a279aea167c106474c2eafe25936b3b6137b96896e344913333c9daa1e6599，发布仅归一22个Shell文件换行。原训练run_cmd已备份至`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/B-E-T0-RA-SMILES-test/previous_run_cmd.sh`；同目录保存sample.out、sample.err及随后evaluate.out、evaluate.err。通过bash -n后仅消费父目录try_lock，after_lock及所有旧产物保留。测试launch名称为formal_B-E-T0-RA-SMILES_test_0f09526。

训练完成核查（2026-09-16 01:12）：stdout报告31200更新、stop_reason=plateau并正常退出；完整last.ckpt确认decline_count=3、last_validation_step=31200、best21600及上述val/loss。最终优化器lr=8e-7，W&B最后训练批次lr=4e-6来自第三次下降前，下降后未再更新参数。旧源码、39份定期checkpoint、last、训练配置及W&B全部保留，没有OOM降批。

测试入口提交为`0f09526ea210b573071c1c601881c16412b7999a`。主代理逐字段自查与独立代理一轮审查批准，未修改生产Python或重跑GPU预实验。本机两个Python环境均缺少PyYAML，因此改用服务器既有运行环境只读解析内存中的新YAML；与已完成的中心复验配置比较，公共数据路径一致，仅模型路径、名称和E协议不同；实际保存训练配置的`data.dataset.pocket_mode=envelope`、RA及检查点路径均核对通过。第一次核查脚本误读顶层dataset，修正为data.dataset后通过，未启动模型或写入测试产物。

验收与资源命令独立于上述正式入口：`tmp/formal-execution-20260914/read_completed_envelope.sh`只读检查完整checkpoint；`start-envelope-test-379402.sh`负责锁控制与留存正式命令。01:14核对实际JobId379402、数组索引0、gnode09、16CPU、A800 UUID GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b；控制器2383等待，显存5MiB。after_lock和父目录try_lock存在、kill_lock不存在，新测试产物目录尚不存在。

2026-09-15 16:58:48（gnode09）备份原控制命令后仅消费父目录try_lock，after_lock和所有中心模型产物保留。正式命令为`bash 训练与运行/sh/train_docking.sh B-E-T0-RA-SMILES`，72×1、bf16-mixed、原E训练与val/loss，从规定官方参数新初始化优化器和调度状态。没有传resume，也没有修改已验收科学配置。

冻结release为`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_e6173e5c817d/PocketXMol`（来源0c79d53）。本次控制脚本为`tmp/formal-execution-20260914/start-envelope-379402.sh`，日志与原命令副本在`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/B-E-T0-RA-SMILES/`，分别为train.out、train.err、previous_run_cmd.sh。实际launch为`/home/penghongen/Feedback/PocketXMol/launches/379402/formal_B-E-T0-RA-SMILES_0c79d53`，已由正式stdout确认；控制脚本属于资源证据，不与正式短训练命令混记。

## 之前的准备与尝试

以下为旧分工下尚未启动正式复验时的准备记录，其中378693接管Matcher是当时已发生的事实；当前职责已改为379402。旧编码实验结果继续留在[原历史日志](../第一类实验（不加密度信息）/3-B-E-T0-RA.md)。

最新执行边界：前置集成及验收收口后先汇报，正式复验等待用户再次明确允许；本次新运行尚未启动。
**当前：修正配体芳香键编码的复验已获准，前置验收完成，等待用户再次允许；尚未开始训练。** 本次从官方pxm权重重新初始化模型、优化器和调度状态，采用公共精确SMILES图，不恢复旧模板编码的checkpoint。资源为378693／gnode10／A800／16CPU；训练bf16-mixed，推理官方FP32张量路径。

| 当前复验字段 | 已确定内容 |
|---|---|
| 配置 | configs/docking/B-E-T0-RA-SMILES.yml |
| 训练产物根 | /storage/penghongen/PocketXMol/training/B-E-T0-RA-SMILES |
| 测试协议 | E；每实例每协议50候选、100步 |
| best／W&B／测试指标 | 尚未产生；旧值见后半部分，不充当本次结果 |

## 正式运行命令

用户再次明确允许后执行的短入口为 `bash 训练与运行/sh/train_docking.sh B-E-T0-RA-SMILES`，当前未执行。训练结束读取实际best后分别记录采样和评价命令，不生成完整验证集候选。

## 验收与资源控制记录

2026-09-14 19:52，主代理在gnode10确认378693正在执行Matcher的smiles_identity.experiment，进程组13179，随后按授权创建实际kill_lock。控制器已终止该进程组并回到try_lock；after_lock保留，GPU0空闲。接管证据根为/storage/penghongen/tmp/pxm_formal_smiles_20260914/takeover-378693。接管不删除Matcher文件或checkpoint，原运行命令另有副本。新链验收命令后续记录于[SMILES日志](../预实验（一 --二之间）/1-SMILES构图与GPU验收.md)。

启动确认（17:03）：stdout核对官方初始权重与本次launch，stderr确认bf16-mixed、在线W&B，31次更新损失有限。控制器准备运行副本约3分钟后正常启动，不属于模型I/O退化，未改变底层控制器或其他项目。沿用原空E口袋跳过规则，进入60分钟分段等待。

核查（2026-09-15 18:05／18:07）：B-E-T0-RA-SMILES：4048更新，best2400／约1.81572，last4000；lr=1e-4，W&B 2uv4mdnt online，尚无测试。三项训练持续产生有限损失，累计OOM均0；在线W&B持续记录。D1测试正常生成，无失败实例，未见明确I/O退化。继续60分钟分段等待，不改科学配置。

核查（2026-09-15 19:07／19:09）：B-E-T0-RA-SMILES：8113更新，best5600／约1.77278，last8000；lr=1e-4，W&B 2uv4mdnt online，尚无测试。三项训练持续产生有限损失，累计OOM均0；在线W&B持续记录。D1测试正常生成，无失败实例，未见明确I/O退化。继续60分钟分段等待，不改科学配置。

核查（2026-09-15 20:09／20:11）：B-E-T0-RA-SMILES：12163更新，best10400／约1.72044，last12000；lr=1e-4，W&B 2uv4mdnt online，尚无测试。三项训练损失有限、累计OOM均0；W&B保持online。D1的C0已全部生成成功，C5正常推进。未见明确I/O退化，继续原配置并分段等待60分钟。

核查（2026-09-15 20:38）：B-E-T0-RA-SMILES：14040更新，最近完整检查点13600；上次核验best10400／约1.72044，lr=1e-4，尚无测试。续接时已核对D1测试、D2中心与包络训练及无密度包络训练进程均存在；未重启任何任务，继续原定等待。

核查（2026-09-15 21:09／21:10）：B-E-T0-RA-SMILES：16009更新，best14400／约1.65910，last16000；lr=1e-4，W&B 2uv4mdnt online，尚无测试。三项训练损失有限、累计OOM均0，W&B均在线。D1 C0全部成功，C5持续推进。D2最近两小时更新吞吐接近，未见明确I/O退化；继续原配置与60分钟分段等待。

核查（2026-09-15 22:10／22:11）：B-E-T0-RA-SMILES：19960更新，best14400／约1.65910，last19200；首次下降后lr=2e-5，W&B 2uv4mdnt online，尚无测试。三项训练持续推进、损失有限、累计OOM均0；W&B在线。无密度包络已首次降学习率至2e-5，按原规则继续；D1测试无生成失败。吞吐未见明确退化，继续60分钟分段等待。

核查（2026-09-15 23:10／23:12）：B-E-T0-RA-SMILES：23842更新，best21600／约1.52456，last23200；lr=2e-5，W&B 2uv4mdnt online，尚无测试。三项训练损失有限、累计OOM均0、W&B在线。D1 C0已完成、C5生成全部成功；既有脚本将在两个协议推理完成后自动开始CPU评价。未见明确I/O退化，继续60分钟分段等待。

核查（2026-09-16 00:11／00:13）：B-E-T0-RA-SMILES：27780更新，best21600／约1.52456，last27200；第二次下降后lr=4e-6，W&B 2uv4mdnt online，尚无测试。D1中心C0/C5各446实例、22300候选均生成成功，2026-09-15 23:34:54进入本资源8进程CPU评价，当前8个子进程均约99% CPU。三项训练损失有限、无OOM、W&B在线；无密度包络第二次降学习率至4e-6。继续原流程与60分钟分段等待。

核查（2026-09-16 01:12／01:13）：B-E-T0-RA-SMILES：训练完成：31200更新、第三次下降停止；best21600／1.5245567560195923，W&B 2uv4mdnt online finished；正在发布E测试入口。无密度包络训练正常退出：31200更新、第三次下降、best21600，完整检查点与在线W&B结束状态已核验；新E测试入口经主代理自查和一轮独立审查批准，正在发布。D1 C0评价完成，Top-1为58.52018%，C5评价继续；D2中心与包络继续在线训练、无OOM。

核查（2026-09-16 01:20）：B-E-T0-RA-SMILES：训练完成，best21600／1.5245567560195923；E测试3/446实例、150候选成功，评价尚未开始。D1包络已从官方参数初始化，W&B n7awsoa3 online，初始数据加载中；无密度包络E测试前三个实例全部生成成功。正式进程已核对，继续60分钟分段等待，不重复启动。

核查（2026-09-16 02:20／02:22）：B-E-T0-RA-SMILES：训练完成、best21600／1.5245567560195923；E测试130/446实例、6500候选成功，CPU评价尚未开始。三项训练损失有限、累计OOM均0，W&B保持online。D1包络已通过初始加载并保存首个验证检查点；D2中心第二次降学习率后继续；无密度包络E测试无生成失败。没有观察到明确I/O退化，继续60分钟分段等待。

核查（2026-09-16 08:52／08:54）：B-E-T0-RA-SMILES：全流程完成，best21600；E测试446实例、22300候选生成和评价成功，Top-1=63.00448%，评价W&B t5ckbgr9 online_completed。两次可见核查之间远端任务持续执行，未进行重启；本次按真实节点时间记录进度。无密度包络05:58:22已完成全流程、评价W&B t5ckbgr9 online_completed；官方冻结对照08:54:38派发。三项训练无OOM，损失有限。W&B只读API本次响应缓慢，尚在等待，未切换offline或认定训练停止。
