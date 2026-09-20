# SciDoc 项目文档

## 2026-09-20 论文协议审计更新

公开 submission 的验证、计分和命令以 [evaluation/README.md](../../evaluation/README.md) 为准。
论文主指标是语义 Answer Accuracy、逐题宏平均 E-Precision / E-Recall / E-F1 和 A-Pages；
精确页集合匹配与联合正确性只是审计诊断。`src/pku_qa/evaluation/` 已移除规则优先匹配，Judge 统一使用论文原版提示词、
严格标签解析和原始输出；内部报告仍不能据此宣称复现当前论文。历史 v4 重聚合匹配 Table 2 的
99/99 个显示值、Table 3 的 98/99 个显示值；新二分类 Judge 的正式历史运行配置仍未恢复。
差异及完整证据见 [official evaluation 审计](../reports/official_evaluation/FINAL_REPORT.md)。
本次整理只读验证 QA；问题、答案、证据、元数据和 manifest 均不修改。


当前发布、人工审核和正式评测统一使用 `data/qa/7.final_2200/` 的四文件集合：
普通 1,000、不可回答 200、Reasoning 200、Cross-PDF 800，共 2,200 条简答 QA。
全局 ID 为 `QA0001`–`QA2200`；最终评测使用 474 份单论文 PDF 和 238 份合并 PDF。

项目介绍与 Quick Start 提供 [简体中文](../../README.zh-CN.md) 和 [English](../../README.md) 两个版本；数据规模、命令和协议说明保持一致。

从 [README Quick Start](../../README.md#quick-start) 开始，无 GPU 的首次数据检查为：

```bash
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200
```

该入口仅需 Python 标准库，不依赖外部 PDF、模型、上游基线或分类镜像，也不改写任何 QA。
添加 `--check-pdfs` 后使用 pypdf 检查最终 PDF 哈希、可读性及证据页范围。

| 需求 | 文档 |
| --- | --- |
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
