# QA 数据集目录

本目录由 `scripts/generate_dataset_catalog.py` 根据当前 `data/qa/**/*.json` 自动生成。
网页“数据集与人工校验”会同步展示同一批可识别为 QA 的文件；位于 `data/qa/` 的模型输出、
断点和汇总材料若仍含 QA 结构，也会列出但明确标为只读过程产物。`data/web/` 中的账户、
审核事件、撤销快照和任务状态不参与数据集扫描。总数、哈希和更新时间
以网页/API 返回为准。不要把本表中的每个 JSON 都当作正式发表输入。

当前识别到 **20** 个 QA-shaped JSON，各文件 QA 记录数相加为 **25011**（含重复谱系和镜像，非独立题数）。
最终 benchmark 仅为 `7.final_2200/` 下四个 QA 文件的 **2,200** 条。其余条目不增加发布规模。

| 文件 ID | 相对路径 | QA / PDF | 状态与类别 | 来源/获得方式 | 用途 | 权限 |
|---|---|---:|---|---|---|---|
| `data/qa/2.unanswerable/rel__single_pdf__unanswerable__batch01__n200.json` | `data/qa/2.unanswerable/rel__single_pdf__unanswerable__batch01__n200.json` | 200 / 132 | 最终集只读镜像；不可回答单 PDF QA | 从 data/qa/7.final_2200/unanswerable_qa.json 按字节同步。 | 单独归档的最终不可回答题镜像；内容必须与 unanswerable_qa.json 完全一致。 | 只读：只读镜像；修改必须在 7.final_2200/unanswerable_qa.json 中完成。 |
| `rel__single_pdf__ordinary__batch01__n1000.json` | `data/qa/1.base/rel__single_pdf__ordinary__batch01__n1000.json` | 1000 / 398 | 最终集只读镜像；普通单 PDF QA | 从 data/qa/7.final_2200/ordinary_qa.json 按字节同步。 | 按用途归入 base 的最终普通题镜像；内容必须与 ordinary_qa.json 完全一致。 | 只读：只读镜像；修改必须在 7.final_2200/ordinary_qa.json 中完成。 |
| `cross_pdf_qa.json` | `data/qa/7.final_2200/cross_pdf_qa.json` | 800 / 238 | 发表前最终人工校验；跨论文多文档推理 | 合并两批各 400 条的严格复审结果，并统一全局 QA ID。 | 最终 2,200 条中的 800 条 Cross-PDF QA，使用规范 z_cross_ PDF ID。 | 可人工修改 |
| `reasoning_qa.json` | `data/qa/7.final_2200/reasoning_qa.json` | 200 / 87 | 发表前最终人工校验；单论文高难推理 | 合并两批各 100 条的严格双审与难度校准结果，并统一全局 QA ID。 | 最终 2,200 条中的 200 条单论文多步推理 QA。 | 可人工修改 |
| `unanswerable_qa.json` | `data/qa/7.final_2200/unanswerable_qa.json` | 200 / 132 | 发表前最终人工校验；不可回答 QA | 由已完成人工终审的单论文正式组件按精确 Unanswerable 规则拆分并统一全局 QA ID。 | 最终 2,200 条中答案精确为 Unanswerable、证据页为空的 200 条单论文 QA。 | 可人工修改 |
| `ordinary_qa.json` | `data/qa/7.final_2200/ordinary_qa.json` | 1000 / 398 | 发表前最终人工校验；普通可回答 QA | 由已完成人工终审的单论文正式组件按精确 Unanswerable 规则拆分并统一全局 QA ID。 | 最终 2,200 条中的 1,000 条普通可回答单论文 QA。 | 可人工修改 |
| `work__reasoning__historical_clean__batch00__n100.json` | `data/qa/3.reasoning/work__reasoning__historical_clean__batch00__n100.json` | 100 / 68 | 历史正式输入；结果追溯 | 旧 Reasoning 正式输入，保留用于复现实验和核对历史结果。 | 刷新前的第一批 Reasoning 100；其中 98 条后来经 PDF 依赖审计替换。 | 可人工修改 |
| `rel__reasoning__refreshed__batch01__n100__v1.json` | `data/qa/3.reasoning/rel__reasoning__refreshed__batch01__n100__v1.json` | 100 / 45 | 历史构建来源；单论文高难推理 | 由旧 Reasoning 100 与严格双审候选按 209 条选择池确定性刷新。 | 对旧第一批替换 98 条并保留 2 条后的历史 Reasoning 100，已完成 PDF 依赖和证据页审计。 | 可人工修改 |
| `data/qa/3.reasoning/rel__reasoning__refreshed__batch01__n100__claude_hard__v1.json` | `data/qa/3.reasoning/rel__reasoning__refreshed__batch01__n100__claude_hard__v1.json` | 100 / 43 | 历史生成谱系；五文件迁移前源组件 | reasoning_qa.json 的迁移前第一批组件。 | 保留用于追溯 reasoning_qa.json 的第一批来源。 | 只读：迁移前源组件仅用于生成谱系；正式审核使用 reasoning_qa.json。 |
| `rel__reasoning__incremental__batch02__n100__v1.json` | `data/qa/3.reasoning/hard_expansion/rel__reasoning__incremental__batch02__n100__v1.json` | 100 / 57 | 历史构建来源；单论文高难推理扩展 | 从同一 209 条严格候选池选择且不与刷新第一批重复。 | 严格 Claude/Gemini 双 KEEP、证据恢复和 4B/8B 闭卷依赖筛选后的新增 Reasoning 100。 | 可人工修改 |
| `data/qa/3.reasoning/hard_expansion/rel__reasoning__incremental__batch02__n100__claude_hard__v1.json` | `data/qa/3.reasoning/hard_expansion/rel__reasoning__incremental__batch02__n100__claude_hard__v1.json` | 100 / 56 | 历史生成谱系；五文件迁移前源组件 | reasoning_qa.json 的迁移前第二批组件。 | 保留用于追溯 reasoning_qa.json 的第二批来源。 | 只读：迁移前源组件仅用于生成谱系；正式审核使用 reasoning_qa.json。 |
| `data/qa/4.cross_pdf/hard_expansion/rel__cross_pdf__strict_dual_review__batch02__n400__v1.json` | `data/qa/4.cross_pdf/hard_expansion/rel__cross_pdf__strict_dual_review__batch02__n400__v1.json` | 400 / 141 | 历史生成谱系；五文件迁移前源组件 | cross_pdf_qa.json 的迁移前第二批组件。 | 保留用于追溯 cross_pdf_qa.json 的第二批来源。 | 只读：迁移前源组件仅用于生成谱系；正式审核使用 cross_pdf_qa.json。 |
| `data/qa/4.cross_pdf/challenge/rel__cross_pdf__challenge__batch01__n400__v1.json` | `data/qa/4.cross_pdf/challenge/rel__cross_pdf__challenge__batch01__n400__v1.json` | 400 / 121 | 历史生成谱系；五文件迁移前源组件 | cross_pdf_qa.json 的迁移前第一批组件。 | 保留用于追溯 cross_pdf_qa.json 的第一批来源。 | 只读：迁移前源组件仅用于生成谱系；正式审核使用 cross_pdf_qa.json。 |
| `data/qa/3.reasoning/calibration/reasoning_claude_wrong_candidates_11.json` | `data/qa/3.reasoning/calibration/reasoning_claude_wrong_candidates_11.json` | 11 / 11 | 过程候选；候选/原始 QA 数据 | 由项目数据目录自动发现：data/qa/3.reasoning/calibration/reasoning_claude_wrong_candidates_11.json。文件内每个顶层记录包含论文或 PDF 标识及其 QA 字段；未登记为显式正式集的文件，其来源关系以该文件所在目录和 annotation_provenance 为准。 | 作为后续清洗、复审或正式集构建的输入，不代表已发布质量。 | 只读：该文件属于模型输出、审核记录、断点或汇总过程材料，网页仅提供查看，避免误改可复现产物。 |
| `rel__single_pdf__short_answer_unanswerable__n4451__v1.json` | `data/qa/1.base/rel__single_pdf__short_answer_unanswerable__n4451__v1.json` | 4451 / 693 | 困难派生；拒答能力评测 | rel__single_pdf__short_answer__n4211__v1.json + work__single_pdf__unanswerable_pool__n240.json。 | 在全简答 4211 题基础上追加 240 道不可回答题，用于同时考察作答能力和正确拒答能力。 | 可人工修改 |
| `rel__single_pdf__short_answer__n4211__v1.json` | `data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json` | 4211 / 693 | 历史构建基线；上游简答基线 | 由 rel__single_pdf__mixed__n4211__v1.json 转换得到；用于构建谱系；当前正式评测使用 7.final_2200/ 四文件。 | 将清洗混合集中的选择题全部转换为简答题，统一使用短答案生成与判卷。 | 可人工修改 |
| `rel__single_pdf__mixed__n4211__v1.json` | `data/qa/1.base/rel__single_pdf__mixed__n4211__v1.json` | 4211 / 693 | 历史构建基线；混合题型评测 | 由原始 6204 题清洗得到，也是全简答 4211 集的直接来源。 | PDF 评测清洗后的 4211 题标准集，保留选择题和简答题两类格式。 | 可人工修改 |
| `work__single_pdf__raw_mixed__n6204.json` | `data/qa/1.base/work__single_pdf__raw_mixed__n6204.json` | 6204 / 703 | 原始数据；溯源基线 | 最上游原始版本，是 option-shuffled 和 4211 标准集的来源。 | 703 篇论文的原始混合题型数据，包含选择题与简答题；尚未经过后续标准集的清洗和筛选。 | 可人工修改 |
| `rel__human_reviewed__authority__batch01__n983.json` | `data/qa/5.human_reviewed/rel__human_reviewed__authority__batch01__n983.json` | 983 / 475 | 历史构建来源；人工审核与同步来源 | 最终四文件形成前的人工审核来源记录。 | 保留供历史单 PDF 同步工作流使用；不代表最终 2,200 条的独立发布输入。 | 可人工修改 |
| `work__single_pdf__unanswerable_pool__n240.json` | `data/qa/1.base/work__single_pdf__unanswerable_pool__n240.json` | 240 / 40 | 辅助题源；拒答压力测试 | 独立辅助题源，合并进入 rel__single_pdf__short_answer_unanswerable__n4451__v1.json。 | 40 篇论文上的额外不可回答题，用于构建 Hard 数据集并评测模型是否会编造答案。 | 可人工修改 |
