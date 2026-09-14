# Docking 共同数据与内存输入

本目录把已有 AdaLigand 实例接入原 PocketXMol 的 free docking 路径。原实例身份、完整模板重原子和世界坐标始终保持一致；训练只增加必要的手性对称排列，密度辅助标签从已有 `ligand_area.npz` 按实例拆出。

下列结构由当前准备代码生成，共同资产已在服务器完成准备；实际数量和核查证据见 [共同准备记录](../日志/实现与共同数据准备.md)。默认路径由 `configs/docking/prepare.yml` 指定。两个目录树中的 `data` 是同一个物理目录，分别列出稳定接口与准备诊断。

## <科学产物>

```text
/storage/penghongen/PocketXMol/data/
├── train.jsonl                 # 原训练质量清单中累计通过的完整实例
├── validation.jsonl            # 原验证质量清单中累计通过的完整实例
├── calibration.jsonl           # 校准实例，始终不并入训练
└── test.jsonl                  # 指定测试 PDB 的累计通过实例及三视图归属

/storage/penghongen/Adaligand_Build/Ori_Data/pocketxmol/
├── symmetries/
│   └── CCD_GMP.npz             # 示例身份 CCD:GMP 的手性自同构排列；每身份一个 NPZ
└── ligand_area/
    └── 9v7o/
        └── 0.npy              # 示例 occurrence 0 的源 ZYX 体素索引；每实例一个 NPY
```

文件名中的 `CCD_GMP` 来自完整 `object_key="CCD:GMP"`，仅把冒号替换成下划线。`candidate_id` 就是项目中的 `occurrence_id`，例如 `9v7o/0.npy`；不能用 CCD 编号替代 PDB 内的实例编号。

## <其他文件>

```text
/storage/penghongen/PocketXMol/data/
├── README.md                   # 当前字段契约的副本（运行说明）
├── summary.json                # 四个划分、三视图及排除原因计数（运行统计）
├── excluded.jsonl              # 每个排除实例的首个失败原因（运行诊断）
└── preparation/
    ├── sources.jsonl           # 原划分与身份条件确定的候选实例（计算中间文件）
    ├── index_excluded.jsonl    # 身份条件失败记录（运行诊断）
    ├── objects_{0..11}.jsonl   # 12个数组分片的模板准备状态（运行诊断）
    ├── samples_{0..11}.jsonl   # 12个分片累计通过、尚未冻结偏移的实例（计算中间文件）
    └── excluded_{0..11}.jsonl  # 12个分片的资产失败记录（运行诊断）

/storage/penghongen/Adaligand_Build/Ori_Data/pocketxmol/
└── README.md                   # 与data目录相同的字段契约副本（运行说明）
```

`{0..11}` 表示正式配置中的 12 份文件。开发验收可以减少分片数，末次 `freeze` 必须使用该次实际分片总数。中间文件不能代替四份最终清单作为训练或测试输入；目录存在也不表示准备完成。

## 完整实例和测试视图

### `train.jsonl`、`validation.jsonl`、`calibration.jsonl`、`test.jsonl`

每行对应一个合法完整实例，由 `freeze` 阶段写出。训练、验证、校准先继承 `stage1_preparation_box_pool_3/split/` 中的原实例资格；测试只使用指定 `held_out_06_chain/test_0.json` 的 PDB。所有划分累计要求原 `small_molecule`、单残基单 CCD、非共价、含碳、原模型词表支持、图与语言资产有效、沉积重原子全部存在且有限、必要受体与地图资产可用。

| 字段 | 类型与实体含义 | 构造示例值 |
|---|---|---|
| `split` | 字符串，所属划分 | `"test"` |
| `pdb_id` | 字符串，源 `parse/` 与 `density/` 的 PDB 子目录名 | `"9v7o"` |
| `candidate_id` | 整数，PDB 内稳定 occurrence 编号 | `0` |
| `object_key` | 字符串，完整共享模板身份，不跨 CCD 合并 | `"CCD:GMP"` |
| `n_heavy_atoms` | 整数，完整模板、沉积坐标和模型配体共同原子数 | `20` |
| `center_offset_xyz_A` | 长度 3 的浮点列表，冻结 C5 的世界 XYZ 偏移，单位 Å | `[1.0, 0.0, 0.0]` |
| `sampling_seed` | 整数，逐实例候选生成种子，同一实例的模型／协议共用 | `10831` |
| `views` | 字符串列表，测试视图归属；非测试为 `[]` | `["ALL", "CAP10", "HF10_TO5"]` |

偏移方向在球面均匀，半径独立服从 `Uniform(0,5)`，不是在球体积内均匀采样。当前C5评测只读取现有冻结值，不重新生成；C0不施加该字段。中心训练与val/loss固定真中心C0，不抽新增偏移。以下只展示格式，不代表正式通过状态或实际冻结种子：

```json
{"split":"test","pdb_id":"9v7o","candidate_id":0,"object_key":"CCD:GMP","n_heavy_atoms":20,"center_offset_xyz_A":[1.0,0.0,0.0],"sampling_seed":10831,"views":["ALL","CAP10","HF10_TO5"]}
```

测试以累计筛选后的完整 `object_key` 计频数。每个身份只建立一次冻结排序：频数大于 10 时，CAP10 取前 10，HF10_TO5 取前 5；频数不超过 10 时两者均全部保留，包括 6–10 的身份。三个视图共用同一候选池、偏移与种子。

### `symmetries/CCD_GMP.npz`

每份文件属于一个共享模板，供训练中的原 `reassign_in` 重新排列等价原子坐标。其内容是 RDKit 对完整图做 `useChirality=True` 自匹配的原规则，不是评价 RMSD 的匹配表。

| 字段 | 类型、形状与含义 | 构造示例 |
|---|---|---|
| `object_key` | NumPy 标量字符串，与源身份完全一致 | `"CCD:GMP"` |
| `atom_count` | int64 标量，完整模板原子数 N | `20` |
| `matches_iso` | int64 `(M,S)`，M 个自同构中会发生置换的 S 个原子列；值索引完整模板原子 | `[[0,2],[2,0]]` 表示模板原子 0 与 2 可交换 |

原 RDKit 匹配最多返回 10000 种排列；如恒等排列未返回则补回。没有可交换原子时形状为 `(1,0)`，不会制造真实扭转或刚体域。原子顺序与 `ligand_objects/CCD_GMP.npz` 的 `atoms` 第一维相同。

### `ligand_area/9v7o/0.npy`

一个实例一份 int32 `(K,3)` NPY，K 为原标签体素数，列为完整源图 Z、Y、X 索引。它直接来自 `density/9v7o/ligand_area.npz` 的 `mask_0`，不会包含其它 occurrence 的监督。

例如源索引 `[25,30,40]` 相对裁块起点 `[20,20,20]` 得到块内 `[5,10,20]`。运行时只保留三个分量都在 `[0,48)` 内的索引。K 可以为 0，此时为 `(0,3)`，不得用其它实例标签填补。实际体素尺寸与源角点从 exp/sim 元数据读取，索引本身没有 Å 单位。

## 准备诊断

### `preparation/sources.jsonl`

每行含 `split`、`pdb_id`、`candidate_id`、`object_key`，是原划分与身份条件筛选后的候选。字段含义与最终清单相同，例如 `{"split":"train","pdb_id":"train_demo","candidate_id":0,"object_key":"CCD:ETH"}`；这时尚未通过完整模板和资产检查。

### `preparation/samples_{0..11}.jsonl`

每行在候选四字段基础上加 `n_heavy_atoms`，例如前述构造实例加 `"n_heavy_atoms":3`。它表示模板、完整性、语言与共同资产检查已通过，尚无冻结偏移和视图；后续 `freeze` 合并所有分片，再写出四份正式清单。

### `preparation/objects_{0..11}.jsonl`

每行一个模板，含 `object_key`、`status`、`reason`、`atom_count` 和 `canonical_smiles`。成功 `status="ok"`、`reason=""`；失败 `status="excluded"`、`reason` 给出异常，后两项为 `null`。`canonical_smiles` 是源字符串的无手性规范形式，仅用于核对已有语言输入身份，不改写源字符串或补算向量。

逐实例语言身份检查读取原 `candidate_<id>.npz`，其中 `model_name` 是标量字符串 `SMI-TED Light 289M`；目录名则为 `smi_ted_289m`。同时核对PDB、实例编号、object_key、已存输入字符串和768维有限向量，不能把目录名当成NPZ中的模型名称。

成功构造示例：`{"object_key":"CCD:ETH","status":"ok","reason":"","atom_count":3,"canonical_smiles":"CCO"}`。失败构造示例：`{"object_key":"CCD:BAD","status":"excluded","reason":"template: unsupported_ligand_bond","atom_count":null,"canonical_smiles":null}`。

### `excluded.jsonl` 与准备阶段的排除文件

每行含实例的四个身份字段与 `reason`。只记录顺序检查中的首个失败原因，不重复扣减；同一 CCD 的图失败会在相关具体实例上体现，不建立额外 CCD 黑名单。身份表丢失时未知 `object_key=""`；测试 PDB 整份身份表不可读时 `candidate_id=-1` 表示无法枚举实例的 PDB 诊断，不能冒充正常实例。

构造示例：`{"split":"train","pdb_id":"train_demo","candidate_id":1,"object_key":"CCD:ETH","reason":"occurrence_assets: incomplete_deposited_heavy_atoms"}`。异常实例直接排除，源资产保持原状。

### `summary.json`

顶层 `source_paths` 保存只读 root、原 split_root、test_split、language_root 及 derived_root；`freeze_seed=3407`、`sampling_seed=10831` 保存冻结随机参数；`splits` 下分别保存四个集合的 `occurrences`、`pdbs`、`objects` 和 `views` 计数，例如 `{"occurrences":27,"pdbs":1,"objects":3,"views":{"ALL":27,"CAP10":26,"HF10_TO5":21}}` 是构造验收集合的格式示例。`excluded_reasons` 是失败原因到实例数的映射，例如 `{"occurrence_assets: incomplete_deposited_heavy_atoms":1}`。

准备完成需要所有对象／实例分片成功退出，`freeze` 成功写出四份清单及统计，并通过记录中的数量、来源和真实输入验收。旧审计数量不能当作新分母。

## 内存装配与使用入口

`OccurrenceDataset` 从最终清单读取实例。完整配体图与世界坐标仍直接读原模板和 `ligand_coords.npz`，按完整模板编号建立原半边表示，调用原 `FeaturizeMol` 和 free 任务／噪声变换。

受体排除 UNK 后按完整残基重原子质量中心选袋：中心距离严格小于 15 Å，包络距离任一真值配体重原子严格小于 10 Å。中心模型原点为本次给定中心，包络模型为实际选入受体原子的世界坐标均值。官方只读蛋白；RA 读标准蛋白及 RNA/DNA，空蛋白不伪造节点或原点。

RA的E口袋为空时，`OccurrenceDataset.__getitem__` 在求均值前抛出 `EmptyEnvelopePocketError`，例如 `empty_envelope_pocket: 6j3z/11`。训练及val/loss的迭代流只捕获这一种错误，按划分和实例身份写警告后跳过；训练继续抽样直至组成正常batch，验证沿原路径计算其余有效E的损失。冻结JSONL及其他源资产不改写。正式采样按records逐实例直接索引，已有失败记录保存全部预算候选的preprocess错误，评价仍保留该实例分母；空E不改用配体中心或补入受体。其他数据异常仍直接传播。

`DataModule.setup`按pocket_mode确定中心C0或包络E，训练与原val/loss条件相同。完整配体世界坐标为X*，g=mean(X*)；中心以g选袋并定模型原点，局部目标X*−g的质心为0。原GaussianExplodePrior加噪后直接执行原同构重分配和固定字段恢复，不增加共享平移，不读取center_translation。

旧运行仍保存含center_translation字段的配置。当前续训、采样和评价的严格配置检查不会把已删除字段自动视为等价；需要读取历史运行时，使用其冻结release与原配置。当前入口服务新的获准运行，使用独立产物目录；现有正确T0结果继续有效，不更改其run.json、checkpoint或W&B。

C0/C5仅是评测提供的实际中心不同，两者使用同一T0推理入口。实际中心c用于选袋及原点C，轨迹中固定；首步原纯高斯先验，后续局部预测Z加s*sigma(N)*epsilon，其中s=1−level_dict['pos']。没有中心相关重新加噪，不强制候选实际质心归零；定位后不读取GT中心或GT偏移，最终只加回C一次。

核酸输入为`(P,15)`：元素C/N/O/P四列、A/C/G/U/DA/DC/DG/DT八列、base/sugar/phosphate三列；蛋白位置为0。蛋白输入保持`(P,25)`，核酸位置为0，`pocket_is_nucleic(P,)`对齐共同原子顺序。RA对蛋白与核酸联合构图，独立投影后使用同一个pocket_encoder；当前无独立核酸编码器或given_center_local字段。

糖组分 sugar 包含 `C1'` 至 `C5'`、`O2'` 至 `O5'`；磷酸组分 phosphate 包含 `P/OP1/OP2/OP3` 及旧名 `O1P/O2P/O3P`；其余标准核苷酸原子归 base。旧原子名的星号转为撇号，例如 `C1* → C1'`。官方空蛋白保留原 `pocket_center(0,3)`，不强制补成正常的 `(1,3)`。

训练对合法实例均匀有放回抽样；有限验证按 worker 编号跨步遍历，正常情况不漏尾部。原偶发 OOM 裁批仍单独记录，不用它解释正常遍历缺失。

正式准备入口为 `scripts/prepare_docking.py`，按 `index → objects → samples → freeze` 执行。每 CPU 任务 8 核、最多 12 份并发；正式 Slurm 命令见 `训练与运行/README.md`。`tests/test_docking_data.py` 使用构造资产检查筛选、原子编号、几何与视图；实际检查和正式运行分别记在 `日志/实现与共同数据准备.md`。

官方等价验收见[test_docking_official.py](../tests/test_docking_official.py)，参照本地Git提交`65488cf635c856101dbe703ac97e2f10f58e005c`的真实原先验、信息等级、噪声器、采样循环与解码。历史六模型源码和配置保存在`463d59098bca92b7278839992afd0efc84d5db81`；旧RB/T1运行须以其对应历史源码理解，当前入口只维护RA＋T0和官方蛋白兼容。收口证据与历史结果见[非密度记录](../日志/第一类实验（不加密度信息）/总日志&分析/非密度收口记录.md)。
