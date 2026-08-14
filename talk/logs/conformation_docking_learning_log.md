# 分子构象生成与小分子对接执行记录

本记录依据 `talk/plans/conformation_docking_learning_plan.md` 执行，关系为“实现该学习计划”。本轮覆盖分子构象生成、小分子对接及两项任务共享的可达代码路径；不覆盖肽任务和其他分子设计任务。

## 基线

- 共同基点：`df1cf846711c3ff7e6284f41cb4b180e8947bac7`
- 累计学习分支：`Learn/CUMULATIVE`
- 当前学习分支：`Learn/conformation-docking`
- 基线状态：已跟踪工作区干净；本地论文 PDF 和 `model_weights.tar.gz` 由 `.gitignore` 忽略。
- 远端状态：本轮不 push，不移动 `origin/*` 引用。

## 已确认决定

- 第一轮只研究分子构象生成与小分子对接，不研究肽。
- PocketXMol 的整体能力来源只在总文档集中分析，不在两份任务档案重复展开。
- GNN 模块纳入学习注释；低层数学模块保持通用契约。
- DataLoader 与损失只深入两个目标任务实际使用的部分，共享字段仍按通用语义说明。
- 本轮只建立学习分支，不建立实现分支。
- 不修改可执行逻辑、YAML 键值、测试、默认值和产物契约；YAML 只补充经消费代码核实的中文注释，疑似缺陷只记录，不直接修复。
- Python 学习注释严格遵守 `code-comment-style-cn`，YAML 注释严格遵守 `yaml-config-style-cn`；Markdown 数学采用 MD Human Review 可识别的美元符号定界符。
- 注释范围按目标函数与实际控制流分支划定；共享文件中的无关实现、默认不可达实验分支和遗留辅助代码保持未注释状态。
- 原六提交端点 `4cb073aad2bac4a730aa33afd082130aedde69e4` 仅由 `codex/archive-conformation-docking-overannotated` 留档；重建历史直接以共同基点为父提交。

## 当前进度

- 已建立 goal、`Learn/CUMULATIVE` 和 `Learn/conformation-docking`。
- 已建立计划、执行记录和映射索引。
- 已沿正式训练/采样入口核对两个任务的数据、transform、noiser、默认 GNN、损失、采样、重建和评测路径。
- 已完成统一建模、输入输出、模块结构、指标排序和两份任务档案，并完成文档—代码交叉复核。
- 已从旧端点的最新版注释做函数级与分支级减法，只保留两个任务实际路径及必要共享契约；未完成实验网络、旧肽裁剪、非目标采样器和任务专用辅助分支均未纳入新学习历史。
- 已在目标 Python 文件中按字段、变量、张量、坐标系和边界函数细化中文学习注释；其中纳入 `process/utils_process.py` 与 `utils/parser.py` 的真实 use/docking 输入链。
- 已为训练、benchmark、checkpoint 与 docking 置信度重评估的直接配置补充中文注释，未改变 YAML 解析值。
- 当前候选将在提交前后分别执行静态行为等价、语法、YAML 解析值、Markdown 链接和差异范围验证。
- 学习历史按 13 个连续阅读章节重建：一个文档前置章节，以及输入、任务构造、训练装配、几何先验、任务 noiser、去噪网络、损失、采样核心、采样入口、基础指标、docking 有效性检查和 confidence 排名 12 个代码章节。远端 `origin/*` 引用保持不变。

## 发现与证据

- `conf` 与 `dock` 共用 `ConfTransform` 和 `ConfSampleNoiser` 主逻辑；`DockSamplNoiser` 只把 `from_train` 的任务名切换为 `dock`。任务的模型侧差别主要是口袋上下文是否为空。
- `PMAsymDenoiser.forward` 不接收 task 名、噪声类型、噪声尺度或时间步；模型实际依赖带噪输入、fixed prompt 和口袋条件识别恢复目标。
- 当前默认分子构象和 docking 都采用 `free` 逐原子高斯噪声；训练配置分别以 `0.001` 概率混入 torsional 和 flexible 设置。
- 论文的小分子 docking 协议为每口袋 100 个 pose，但 `configs/sample/test/dock_poseboff/base.yml` 当前为 `num_repeats: 50`。仓库内没有足够证据说明另一半候选如何组成，复现前需要额外实验记录；本轮不改配置。
- 当前 self-ranking 并非纯置信度：`scripts/rank_pose.py` 使用 `cfd_traj + int(no_clashes) + int(stereo)`。完整 PoseBusters 检查在另一评测步骤执行。
- 统一训练配置的 `dock` 数据混合包含 `apep` 和 `pepbdb`。本轮档案仍只定义并评估小分子 docking，但会如实注明 checkpoint 的联合数据来源，不展开肽协议。
- 构象任务以空口袋张量复用同一模型；空上下文经过当前 PyG `knn` 的运行兼容性留待最终环境验证，不据此判定缺陷。
- 当前 checkout 不含配置所指向的 `data/test/...`、`data/poseboff/...` benchmark 文件，因此本轮只能静态核对路径契约，不能执行真实数据采样或指标复现。

## 验证记录

- 对共同基点到当前学习端点间的 27 个修改 Python 文件执行 `ast.parse`，全部语法通过。
- 去除模块、类和函数 Docstring 后逐文件比较 27 个 Python 文件的新旧 AST，未发现任何可执行结构差异；源码改动限于注释、Docstring 和文件末尾空白清理。
- `rg` 未发现仓库通过 `__doc__`、`inspect.getdoc`、`pydoc` 或 `help()` 消费 Docstring。
- 10 个相关 YAML 文件只增加或细化注释，逐文件 `yaml.safe_load` 结果与共同基点完全相同；`tests/` 和 `test/` 相对共同基点均无改动，仓库没有独立 Python 测试文件。
- 9 个 `talk/*.md` 的相对链接均能解析到现有文件；`git diff --check` 无空白错误。
- 默认 Conda `base` 环境可导入 `utils.info_level` 和 `models.sample`；其他目标模块因缺少 `torch_scatter`、`torch_geometric`、RDKit、OpenBabel 或 EasyDict 无法做导入级验证。当前机器没有专用 PocketXMol Conda 环境。
- benchmark 数据路径在当前 checkout 中不存在，因此没有运行真实构象/docking 采样、RMSD、COV/MAT 或 PoseBusters；这与本轮“不复现完整 benchmark”的计划边界一致。

## 计划与实现差异

- 为解释 flexible/torsional 的 `outputs2batch` 边界，额外细化了 `models/corrector.py` 的几何投影与刚体对齐契约；未扩展到其他任务。
- 为闭合真实 use/docking 入口及其预处理、旋转和落盘链，额外纳入 `scripts/sample_use.py`、`process/process_torsional_info.py`、`utils/motion.py`、`utils/reconstruct.py` 与 `utils/buster_tools.py`；这些文件都位于两个目标任务的可达路径或直接辅助路径上。
- 除上述可达辅助模块外，交付范围与计划一致；没有修改运行配置值、默认值、测试、科学公式或输出格式。
