# SciDoc 项目文档

最新评分入口与验证见 [sxz v4 对齐报告](../reports/official_evaluation/SXZ_V4_ALIGNMENT.md)。此前 current-release 二分类分差报告保留为历史记录。

## 2026-09-21 sxz v4 评分规则对齐

按项目负责人要求，`evaluation/` 默认使用生成论文实验结果的 sxz v4 规则：原始三分类
提示词、源脚本预测恢复、结果文件自带历史金标、先证据后答案、固定分母和仅 CORRECT
计分。现行入口为 `evaluation/evaluate.py`；`reproduce.py` 复用同一评分器并比较论文。
这替换此前二分类公开评分规则，不新增模型协议或兼容模式。`sxz/`、QA、PDF、论文不修改。
全量缓存回放匹配历史 77 行汇总的全部 1,540 个数值、Table 2 的 99/99 和 Table 3 的 98/99；
Claude Table 3 All 仍为 68.05% 对论文 69.32%。这是原始缓存回放，不是新 GPU Judge 运行。
命令和输入结构见 [evaluation/README.md](../../evaluation/README.md)，
证据见 [对齐报告](../reports/official_evaluation/SXZ_V4_ALIGNMENT.md)。
内部 `src/pku_qa/evaluation/` 保留二分类诊断，固定读取 `paper_semantic_judge.txt`；
不能将其报告当作 v4 评分。旧 current-release 二分类重评命令已停用，旧报告仅记录当时结论。

当前发布、人工审核和内部新推理使用 `data/qa/7.final_2200/` 的四文件集合：
普通 1,000、不可回答 200、Reasoning 200、Cross-PDF 800，共 2,200 条简答 QA。
全局 ID 为 `QA0001`–`QA2200`；最终评测使用 474 份单论文 PDF 和 238 份合并 PDF。

项目介绍与 Quick Start 提供 [简体中文](../../README.zh-CN.md) 和 [English](../../README.md) 两个版本；数据规模、命令和协议说明保持一致。

新用户从 [中文 Quick Start](../../README.zh-CN.md#quick-start) 或
[English Quick Start](../../README.md#quick-start) 开始：安装轻量评测依赖，运行无资产
自检，取得原实验文件后回放 Qwen3-VL-8B，最后读取报告中的指标与输入绑定。
缓存回放不要求 GPU 或 PDF；新 Judge 运行和当前数据集 PDF 检查有各自的环境要求。

| 需求 | 文档 |
| --- | --- |
| 单模型评分、全部模型回放、输出与退出码 | [evaluation Quick Start](../../evaluation/README.md#quick-start) |
| 理解协议、模块、字段与目录 | [项目架构](ARCHITECTURE.md) |
| 安装环境、预检和运行评测 | [上手指南](GETTING_STARTED.md) |
| 按题执行人工校验 | [人工审核手册](MANUAL_REVIEW_GUIDE.md) |
| 账户、权限与公网部署 | [Web 控制台](WEB_CONSOLE.md) |
| 区分最终四文件与上游、过程材料 | [QA 数据集目录](DATASET_CATALOG.md) |
| 历史证据定位诊断的适用范围 | [报告索引](../reports/README.md) |

内部推理编排由 `pku_qa.workflows.reporting.run_final_2200_evaluation` 按 manifest 构建
四个锁定路径、数量、QA 哈希的 PDF 评测命令；每个组件依次运行 4B、8B、Judge 和严格报告。
`question_only` 用于闭卷对照；Full、Oracle 和消融是 `pdf` 页选择策略。
`Unanswerable` 必须精确拼写，PDF 拒答证据页为空。严格报告不接收不完整或失去绑定的结果。

Reasoning 两批 100、Cross-PDF 两批 400、基础 4,211 和旧单 PDF 1,190 等记录只说明构建
谱系或历史实验。最终组件之间不能再以旧批次文件重建当前人工审核金标准。
单 PDF 专项实验使用最终普通与不可回答题生成 1,200 题合并输入和动态消融输入。

论文 Dataset 与附录材料保存在 `论文/写作材料/`；该目录包含中文草稿、英文参考稿和图表
生成代码。论文中的统计以最终四文件、发布 manifest 和可复算分类工作簿为依据。
图表生成器从最终 JSON 重算统计，只生成、校验和同步图片资产，不读取、重建或回写
Word 草稿。人工审核界面和证据 PRF 示例使用附录编号，保存在
`论文/写作材料/图/附录/`；完整运行默认另将图片副本同步到 LaTeX 图目录，
`--figures-only` 禁用该复制步骤。图 3-3 展示最终集合的题型、推理、证据页数和模态统计。
图 3-2 改为等宽三栏，与正文三个构建小节同名，删除统计栏；统一卡片标题 11.5 pt、正文 10.5 pt，单条说明最多三行。生成器的 `--pipeline-only` 仅重绘该图。
三栏标题的两行文字分别居中；`--pipeline-only` 默认同步更新
`论文/ICLR2027_ScienceDoc/figures/fig_3_2_sci_doc_pipeline.png`（正文 Figure 2）。
图 3-1 的卡片标题与图标按实际渲染尺寸整体居中，QBIO 标题分两行并检查卡片边界；正文使用 Times New Roman 12 pt，图标按物理尺寸等比绘制。
图 3-1 移除顶部标题、副标题和中心 `source`，降低 CS、PHY 卡片高度约 14%，等比例缩小圆环并收紧画布至 3600 × 2184；生成器的
`--domain-only` 仅重建该图并同步 ICLR 论文图片副本。
图 3-3（当前论文 Figure 4）删除重复的 Evidence Span，保留完整证据页数分布；
最小字号为 10.5 pt，画布为 3552 × 1520，比原图缩短 14.3%。统计卡片的数字与说明
整体垂直居中并留出行间距，堆叠色条内的数值按可见字形边界垂直居中。数字在上述居中位置基础上再下移 5 个输出像素，作细微视觉校正。长题型名称按字宽换行；
展示名称统一为 Task Types、Question Types、Reasoning Types、Document Type、
Scientific Fields 与 Equation，移除 Leading 并采用标题式大小写。
Reasoning Types 的百分比分母仍为 602 条标注记录，不限于 200 道 Reasoning 题；
保留 `Reasoning Types` 标题，六条数据下方以灰色小字标注 `Top 6 of 11 Types`，
样式与 Question Types 的下方注记一致，表示仅展示全部 11 类中的前六类，
共 541 条（约 89.9%），未将部分列表重新归一化为 100%。
`0 (Unans.)` 表示 200 道不可回答题没有金标准证据页。
`build_dataset_figures.py --composition-only` 只重绘本图并默认同步至
`论文/ICLR2027_ScienceDoc/figures/`；正文图注与对应统计段落同步更新术语。
项目维护状态与文件哈希清单属于内部维护资料，不作为论文正文或附录引用。
2026-09-16 已恢复误删的 `论文/写作材料/图/` 中 22 个原文件；恢复来源与逐文件校验值见
[插图恢复记录](../reports/FIGURE_RECOVERY_20260916.md)。该目录仍被 Git 忽略。

附录五个定性案例对应最终发布集 QA0685、QA1830、QA1598、QA1399、QA1323。
`build_case_study_cards.py` 按统一模板生成 600-DPI PNG、独立 draw.io 源文件和五页合集，
并将 PNG 同步到 ICLR 论文图片目录；恢复前备份和逐题核验保存在论文写作材料目录。
2026-09-17 五个案例的红框改为缩放后绘制，统一为 2 个版式像素（600-DPI PNG 中 6 像素）
的细线；标题统一为 `Case N: Document-ID <document ID>    QA-ID <QA ID>`。
生成器同步目标为当前稿件 `论文/ICLR2027_ScienceDoc/figures/`，同时刷新 draw.io 与 Overleaf 包。
案例按精确物理页重算证据指标，区分正式评测与定性运行记录；来源必要性、题目模态和
提示词版本存在的限制需如实说明，不通过修改历史模型输出或金标准来消除差异。

代码或结构变化后运行完整测试：

```bash
python -m pytest -q
```

Web 变更还需 Web build/tests、服务重启和
`PYTHONPATH=src python -m pku_qa.workflows.operations.check_web_stack` 公网验收。
`sxz/` 始终只读。

第一阶段 Notebook 和生成提示词位于 `scripts/dataset_construction/phase1_paper_acquisition/`，用于追溯论文获取与基础 QA 构造；其过程性输出不替代最终四文件。

2026-09-16 论文图稿的类别显示名称统一为 `General` 与 `Multi-Document`，
覆盖当前 ICLR LaTeX 正文、附录、图表生成源及论文插图；数据文件名和评测键不变。
总览图 `fig_3_0_scidoc_overview.png` 改为横向完整论文标题卡片，取消重复短标题，
问答正文由 8 pt 增至 11.5 pt、证据说明为 12 pt，扩大统计图例并分行展示。
修订可通过写作材料中的 `build_dataset_figures.py` 与 `update_paper_figure_labels.py` 重现。

2026-09-18 Figure 1 删除来源文档数分布和 `8 fields` 展示，改用 `Statistics`、
`Modality Composition` 与 `Evidence Modalities (overlapping)`。QA1408 对照原始
合订 PDF 第 9、24、30 页校订：Paper 1 为 −78 μs/yr，Paper 2 的 76.5 μs/yr
为衰减幅度，带符号时约为 −76.5 μs/yr；约 1.96% 的差异仅支持百分比量级的一致性，
不将 Paper 2 的观测/GR 比值精度归给 Paper 1。完整校对依据见
[图表生成说明](../../论文/写作材料/图/图表生成代码/README.md)。
`build_dataset_figures.py --overview-only` 现在默认替换当前
`论文/ICLR2027_ScienceDoc/figures/fig_3_0_scidoc_overview.png`，
`--figures-only` 仍可禁止复制；正式 QA 和历史评测结果不变。

附录 Detailed Dataset Construction, Quality Control, and Evaluation Protocol 的
6 个完整提示词已由 `build_appendix_prompt_cards.py` 提取为可编辑 draw.io 方框。
输出与使用方法见 [提示词图稿说明](../../论文/写作材料/图/附录/提示词方框_drawio/README.md)。
图稿保留原文，预设 Times New Roman，提供六页合集、独立文件和高清 PNG；
6 张图已升级为 5124 像素宽、600 DPI，并替换指定附录节的六个提示词块。
使用原位 minipage，不新增图号；正文和其他附录源内容不在本次替换范围内。
本地预览编译在临时副本中预先传递 `xcolor` 的 `table` 选项，避开现有会议样式的选项冲突；论文导言区未修改。

2026-09-16 已精简 ICLR 稿件的附录 Dataset 和 Detailed Dataset Construction, Quality Control,
and Evaluation Protocol：正文保持不变，附录以正文为准，删除重复统计、图表和模型名单，
保留构建谱系、审核细则、评测公式及六张高清提示词图；五个定性案例保持原样。
正式文件名使用 `ordinary_qa.json` 和 `cross_pdf_qa.json`，语义判分与证据指标分别说明。
手动同步 Overleaf 只需论文 `.tex` 和六张 `figures/appendix_prompt_*.png`；本次文字精简未新增图片。

2026-09-18 ICLR 稿件 Table 4（Construction stages, outputs, and quality gates）
取消整体缩放，使用 9 pt 字体和按页宽自动换行的五列表格，适当增加行距。
表格内容保持不变；同步 Overleaf 只需更新 `论文/ICLR2027_ScienceDoc/iclr2027_conference.tex`。
