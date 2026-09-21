# 实验来源、输入版本与评分差异核对

> **历史记录，现行入口已更新（2026-09-21）：** 本文保留当时的二分类/严格格式诊断、命令和测试结果，
> 不描述现行评分器；文中的旧诊断命令及旧 `--predictions` 评分接口已停用。
> 当前 `evaluation/` 使用 sxz v4 原始提示词与评分规则；55 文件及原始缓存回放匹配
> 历史汇总 1,540/1,540 个数值、Table 2 的 99/99 和 Table 3 的 98/99 个显示值。
> 唯一论文差异是 Claude Table 3 All（68.05% 对 69.32%）。见
> [当前验证结果](SXZ_V4_ALIGNMENT.md)及[可执行 Quick Start](../../../evaluation/README.md#quick-start)。

2026-09-20。本文记录当时核对的结论，补充此前的
[分差调查](GAP_INVESTIGATION.md)，并纠正将当前 release 诊断称为“论文实验不能复现”的过度概括。

## 结论与完成范围

**Table 2/3 有实际历史实验记录支持。当前新判分与历史表格存在差异，不能因此否认历史实验。**
本次完成来源锁定、新适配器复查、Qwen8B 逐题分差归因、Claude 学科汇总人口检查、
必要的本地 Judge 配对实验，以及可复用检查工具和文档修正。

1. 11 模型 × 5 文件，共 **55/55 份** `data/results/` 结果，与历史 v4 manifest
   指向的 `sxz/` 原件 SHA-256 一致。
2. **21,050 条**历史 Judge 缓存的键都与实际问题、参考答案、预测、数据集及提示词哈希一致。
   这不补足历史权重 revision/字节哈希的缺失。
3. 本次直接从逐题工作簿独立重聚合，不依赖旧汇总 CSV 的数值：Table 2 **99/99**
   个显示值一致；Table 3 **98/99**，包含全部 **88/88 个学科值**。
4. 新 evaluator 的 Qwen8B 内容绑定缓存可完整重放；独立整数求和与四任务、八学科汇总一致。
   合法记录的答案字符串逐字等于原始 JSON，证据仅排序去重。本次未发现新的、能解释
   13 个百分点分差的适配或聚合程序错误。此前已修复的“仅金标变化就排除预测”缺陷仍有回归测试。
5. 新 Judge 也存在需要复核的判定；**55.32% 是特定当前输入/配置下的诊断值，不能直接替换论文的 68.32%。**

完整证据在 [provenance_reconciliation/](provenance_reconciliation/)。
论文、原始预测、旧 Judge 缓存、PDF、所有 27 份 QA JSON 均未修改。

## 已执行计划

| 步骤 | 执行与结果 |
| --- | --- |
| 锁定历史来源 | 检查 v4 manifest、55 对原件/副本、工作簿、旧缓存、论文；输出 SHA-256 |
| 复核新 evaluator | 检查独立 gold、原始字段、legacy ID 唯一性、缓存输入绑定、固定分母、任务与学科求和 |
| 拆解 Qwen8B 分差 | 2200 个固定槽位逐题列账；以实际旧缓存输入取代共享工作簿金标作为输入一致性证据 |
| 核查 Claude Table 3 | 找到 28 个清单外正确项、38 个清单内无预测项；逐项查找同 PDF/相同问题对应项 |
| 有限补跑与交付 | 49 对输入、98 次真实本地判分；修正审计方法、报告解释与机器可读状态；测试并推送 |

## 历史实验链路

历史评分入口：
`sxz/evaluate_11models_5datasets_calibrated_fixeddenom_seed42_gpu2345_parallel_v4.py`。
Table 3 构建入口：`sxz/build_table3_by_subject.py`。
结果来源：`data/results/evaluations/detail_11models_5datasets.xlsx` 与
`summary_11models_77rows_calibrated_fixed_denominator.csv`；对应历史原件都在
`sxz/evaluation_11models_5datasets_qwen36_calibrated_fixeddenom_v4/`。

历史 v4 用结果 JSON 自带参考答案和证据，调用 Qwen3.6-27B 三分类 Judge，只有 CORRECT
计正确；解析可读取已解析答案，并恢复部分带 Markdown 等形式的输出。当前论文要求独立
冻结 gold、严格原始 JSON、原版二分类提示词和确定性的规范拒答判断。两者均使用固定分母，
证据主指标均为逐题宏平均。三分类本身不表示实验错误；具体输入、规则、提示词和模型运行
才是核对对象。当前实现没有添加历史评分模式，没有把旧三分类标签导入正式 Judge 缓存。

历史 v4 已记录 seed=42、greedy、32 输出 token、禁用 thinking、本地模型路径，
但未把历史 checkpoint 字节/revision 绑定到缓存。因此本次只能确认**Judge 运行之间的差异**，
不能把全部差异单独归因于提示词、权重或任一配置。不能从目录名断言由哪位同学启动实验。

## Qwen3-VL-8B：286 道净差异完整列账

历史正确 **1503/2200 = 68.31818181818181%**；当前诊断正确
**1217/2200 = 55.31818181818182%**。净差 286 题，即精确 13 个百分点。

| 按真实输入/处理状态分组 | 题数 | 旧正确 | 新正确 | 净减少 |
| --- | ---: | ---: | ---: | ---: |
| 实际 Judge 三个输入字符串完全相同 | 1883 | 1344 | 1112 | 232 |
| 实际 Judge 输入不同 | 49 | 31 | 19 | 12 |
| 旧流程跳过、新流程调用 Judge | 2 | 0 | 1 | -1 |
| 合法 Unanswerable 项 | 188 | 85 | 85 | 0 |
| 当前 raw contract 下非法 | 23 | 11 | 0 | 11 |
| 问题/输入版本无法对应当前 release | 43 | 32 | 0 | 32 |
| 当前 release 缺失预测 | 12 | 0 | 0 | 0 |
| 合计 | 2200 | 1503 | 1217 | 286 |

1883 题中有 234 题旧正确→新错误、2 题旧错误→新正确。这里的“正确/错误”均指
**各自保存的自动判分标签**，不代表已经获得人工裁决。
[逐题 CSV](provenance_reconciliation/qwen8b_items.csv) 保留实际旧缓存键。

此前报告按“release 金标是否变化”分组，不能准确隔离真正送入 Judge 的文本。
例如，仅证据金标变化不会改变语义 Judge 的三个输入；历史空白规范化也可能改变输入字节。
本次从每个模型自己的原始记录和实际旧缓存恢复输入，修正这一审计方法；没有改变任何评分。

### 49 对输入的受控本地检查

使用原版论文提示词、本地 Qwen/Qwen3.6-27B、与此前诊断相同的权重指纹和生成设置：
BF16、SDPA、greedy、seed 42、禁用 thinking、32 输出 token、批量不超过 32、左侧 padding。
这些是明确记录的本次实验设置，不冒充历史二分类运行配置。
只替换 Judge 输入文本，不改变当前 QA 或预测。

- 49 对中，36 对参考文本不同、13 对预测文本不同；包括历史空白处理造成的差异。
- 共 98 次真实判分，**0 失败、0 重试**，含权重校验和加载共 **136.28 秒**。
- 当前输入重复判分 **49/49** 与此前一致。
- 换为历史文本后，40 对标签相同、9 对不同：8 对是参考文本变化，1 对是预测文本变化。
- 在同一论文提示词下，历史文本得 20 个 CORRECT，当前文本得 19 个，净差仅 **1 题**。
- 历史运行原来在这 49 题上得到 31 个 CORRECT；31→20 仍属 Judge 运行差异。

因此，一条明确顺序的差异分解是：Judge 运行差异净减少 243 题，Judge 输入文本变化净减少
1 题，旧跳过项新判分净增加 1 题，非法输出减少 11 题，无法对应当前输入的项减少 32 题；
合计 286。该分解描述观测到的运行差异，不是对哪个 Judge 更准确的估计。
[配对结果](provenance_reconciliation/paired/summary.json) 和原始调用均留档。

### 对新 Judge 判定的质性复核

以下是文本核对观察，不是新的人工 gold，也不覆盖机器判定。
样本按各任务分歧项的 QA ID SHA-256 排序抽取，并包括两个反向分歧；
它们不是全体题目的无偏准确率样本。

| QA | 观察 | 能支持的结论 |
| --- | --- | --- |
| QA1318 | 问题问限制理论有效性的“物理现象”；金标含“resonant hybridization”，模型答“Resonant hybridization”；新 Judge 判错 | 新 Judge 可能把金标补充说明当成必答信息，需复核 |
| QA1299 | 问题问双星团系统的演化来源；模型答一个星团被潮汐力撕裂，与金标的来源一致；新 Judge 判错 | 不能仅因金标还描述统计模型就自动认定模型答错 |
| QA0231 | 金标为“CCC-GARCH with Gaussian innovations”，模型只答“CCC-GARCH model” | 是否遗漏必要限定，需要结合问题和上下文审查 |
| QA0630 | 模型重复“geometrically characteristic subgroup”，未给出金标所要求的不变性性质 | 新 Judge 判错有可解释依据 |
| QA0042 | 模型明确给出较低 FNP、FDP 更接近目标，旧 Judge 判错、新 Judge 判对 | 分歧不是单向的，旧标签也不能视作人工真值 |
| QA1409（此前已核查） | 参考时间为 1.1×10^6 年，模型回答 8.4×10^3 年 | 与论文要求的数值一致性存在明确冲突 |

这些例子说明：**既不能把全部差异说成旧结果错误，也不能为保留旧分数而修改论文提示词、
重写答案或手动覆盖标签。** 当前代码遵守原版提示词，并不意味着每个模型 Judge 决策必然正确。

## Claude Table 3：集合差异，未找到可证实的 ID 修复

历史原始 ordinary 文件有 1190 题。Table 2 直接汇总各组件中已有的判分，
得到 Claude **1525/2200 = 69.3181818%**。
Table 3 按独立的 2200 题学科清单连接，得到 **1497/2200 = 68.0454545%**。
28 个正确项在学科清单外；另有 38 个清单内槽位没有 Claude 预测。缺失项仍留在分母。

对全部 28 项检查：在历史学科清单和当前 ordinary/unanswerable release 中，
均找不到**同一 PDF、问题文本完全相同**的候选。没有证据支持重命名 ID 后补回。
Table 3 的八个学科数值确实对应 1497 个正确项，All 单元格却用了 1525 个正确项的汇总值。
这解释了唯一的 1.27 个百分点显示差异。

本次修正的是审计/报告层：明确输出集合外记录、集合内缺失记录及
`contains_out_of_cohort_predictions`，阻止将集合差异描述为舍入误差或可自动修复的 ID 错位。
未把这 28 题移入当前基准、未给 38 个空槽虚构答案、未改论文表格。
逐项问题和候选匹配见 [audit.json](provenance_reconciliation/audit.json) 的 Claude membership。

## 尚未解决的发布问题及明确处理

- **历史表格来源：已查清。** 不再用新诊断差异否认历史实验。
- **新代码适配/求和：本次核查通过。** 原始字节、当前独立 gold、内容绑定缓存和固定分母没有发现新的计数缺陷。
- **历史 Judge 配置：未完全恢复。** 仍缺执行时权重/revision 的绑定，不能声称已复现相同推理运行。
- **论文文字与历史实现的对应：仍有差异。** 保持论文为标准；没有自动切换到旧提示词或宽松解析。
- **Judge 判定质量：有具体待审案例。** 不能用未经专家裁决的新标签直接替换论文结果。
- **非法输出的证据指标：仍显式 unresolved。** 当前论文未覆盖所有非法 raw 情况的页集合恢复规则。
- **全体 11 模型的新推理/新判分：本次未启动。** 已完成的配对检查足以定位主要差异阶段；
  重跑全部推理不能恢复缺失的历史执行配置，也不能解决对新 Judge 标签质量的疑问。

上述问题涉及实验版本与协议事实，不能通过改几行指标代码保证旧数值不变。
本次交付是完成核对和工具修正；不是宣称所有公开发布条件已经满足。

## 文件、验证与重跑命令

新增 `evaluation/audit_score_provenance.py`、`scripts/check_judge_input_versions.py`、
`tests/test_score_provenance.py` 及本报告/证据；更新 evaluator README、reconciliation 状态、
当前文档入口和四份旧报告的解释说明。`check_reproduction.py` 的结论文字现明确限定为当前
release 诊断，不把条件性上下界检查写成对历史实验真实性的结论。

从仓库根目录运行（使用新输出目录，避免覆盖已有审计）：

```bash
python -m pip install -r requirements-eval.txt
python evaluation/preflight.py --output /tmp/sciencedoc_provenance_preflight.json
python evaluation/audit_score_provenance.py \
  --rescore-dir data/results/qwen8b_current_gold_rescore_20260920 \
  --output-dir data/results/provenance_check_new
python -m pytest -q
```

来源检查需要本机历史原件/工作簿及已保存的完整诊断缓存；这些外部实验资产不保证随 Git 克隆。
在已有 Torch/Transformers 和本地权重的实验环境，可独立重复配对检查（GPU 2 须为空闲 A800）：

```bash
python scripts/check_judge_input_versions.py \
  --audit-dir data/results/provenance_check_new \
  --rescore-dir data/results/qwen8b_current_gold_rescore_20260920 \
  --output-dir data/results/provenance_pairs_new --gpu 2
```

不需要 API key。GPU 用 UUID 绑定和仓库 reservation helper；禁用 CPU/disk offload。
完整测试、预检及不可变文件验证结果见
[verification.json](provenance_reconciliation/verification.json)。
