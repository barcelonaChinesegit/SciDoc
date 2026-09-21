# ScienceDoc evaluation — sxz v4

自 2026-09-21 起，`evaluation/` 的默认评分规则按项目负责人要求，改为生成论文实验结果的
`sxz/evaluate_11models_5datasets_calibrated_fixeddenom_seed42_gpu2345_parallel_v4.py`。
这是现行评分实现的替换，不增加兼容模式或第三种模型输入协议。`sxz/` 始终只读。

`evaluation/sxz_v4.py` 逐字保留原脚本的解析、Judge、逐题评分与固定分母汇总函数；
`prompts/semantic_judge.txt` 是原始 `JUDGE_RULES` 字符串，完整提示词还包含原脚本的
外层指令、Dataset、Question、Gold Answer 和 Model Prediction。来源与逐函数哈希在
[prompts/provenance.json](prompts/provenance.json)。唯一移植差异是本地权重路径从当前
checkout 解析；原脚本的输出目录和 GPU 调度器未复制。

**全量验证：55 份结果经这套代码和原始 Judge 缓存重新计算，77 行汇总的 1,540 个数值
全部与历史汇总一致；论文 Table 2 为 99/99，Table 3 为 98/99。**
Qwen3-VL-8B 的 All 为 68.32%。唯一论文差异仍是 Claude Table 3 All：原学科集合汇总为
68.05%，论文为 69.32%。不会为了匹配该单元格而更改样本集合或正确标记。
验证证据见 [sxz v4 对齐报告](../docs/reports/official_evaluation/SXZ_V4_ALIGNMENT.md)。

## Quick Start

以下命令从仓库根目录执行。独立评测包需要 Python 3.10+；新克隆的安装方法见
[项目 Quick Start](../README.zh-CN.md#quick-start)。

### 1. 安装与无资产自检

```bash
python -m pip install -r requirements-eval.txt
python evaluation/evaluate.py --help
python -m pytest -q tests/test_sxz_evaluation.py tests/test_judge.py tests/test_e2e.py tests/test_metrics.py
```

不需要模型、PDF、GPU、网络凭据或历史结果；干净克隆缺少 `sxz/` 原脚本时，该原件
对照测试会跳过。其余随包提供的规则测试仍运行。

### 2. 最小真实回放

先取得 `data/results/Qwen3-VL-8B/` 下的五个原始结果文件，以及下列原始 Judge 缓存。
这些材料不在 Git 中。缓存回放不读取 PDF，也不重新调用模型。

```bash
python evaluation/evaluate.py \
  --results-dir data/results --model Qwen3-VL-8B --offline \
  --judge-cache sxz/evaluation_11models_5datasets_qwen36_calibrated_fixeddenom_v4/qwen36_answer_judge_cache.json \
  --output-dir data/results/quick_start_qwen8b
```

原始材料的预期结果：退出码 0、Table 2 匹配 9/9、`paper_differences: []`。
`report.json` 中的 `table2["Qwen3-VL-8B"]` 给出 All=68.32%、E-F1=40.66%、A-Pages=8.44
（显示到两位小数）。逐题结果在模型子目录的 `details.csv`。

### 3. 按需求继续

- 全部 11 模型、Table 3 和历史汇总对照：[完整回放](#原实验缓存回放无需-gpu)。
- 用当前本地权重重新判分：[新本地 Judge 运行](#新本地-judge-运行)。
- 数据和严格生成格式检查：`preflight.py` / `scripts/validate_submission.py`，
  见 [当前数据与格式检查](../README.zh-CN.md#当前数据与格式检查)。它们需要发布集 PDF，
  不属于上述缓存回放的前置步骤。

## 评分规则

| 环节 | 与 sxz v4 相同的行为 |
| --- | --- |
| 金标 | 每个结果 JSON 自带的 question、answer、evidence_pages；不替换成当前 release 金标 |
| 答案恢复 | 先读已解析预测字段，再读原始生成字段；允许代码围栏、JSON 片段、嵌套容器等源脚本支持的形式 |
| 文本 | 按原脚本压缩空白；不使用别名匹配、数值容差或字符串相似度预判答案正确 |
| 证据恢复 | 按原脚本字段优先级及 raw fallback 提取页号，集合去重；不额外加入当前严格页类型/页上界门禁 |
| 证据评分 | 先算证据，再检查答案是否可恢复；错误答案仍可获得证据分，非法输出不再造成全体证据指标 null |
| 空集合 | 金标和预测均为空时 P/R/F1=1；仅一方为空时为 0；缺失金标字段不等于空集合 |
| Judge | Qwen3.6-27B 原始校准提示词：参考答案不是逐项检查清单，问题要求的必要内容才必须回答 |
| 标签 | CORRECT / PARTIAL / WRONG；**只有 CORRECT 计正确，PARTIAL 不给半分** |
| 拒答题 | 与历史实验相同，也调用 Judge；模型生成规范仍使用 `Unanswerable` |
| Judge 回复解析 | 原脚本正则、大小写及优先级；不能解析的回复中止运行，不计为模型答错，不自动重试 |
| 缺失结果 | 固定分母保留，贡献零；无法恢复的预测记技术失败，已恢复证据照常计算 |

原始结果文件保持不变，报告记录结果文件、提示词、代码、缓存和学科清单哈希。
新调用的缓存与运行身份绑定；旧二分类 JSONL 缓存不能续跑。

## 输入与安装

```bash
python -m pip install -r requirements-eval.txt
```

每个模型目录需要原实验的五个文件：

```text
RESULTS_DIR/MODEL/ordinary1190.json      # 固定分母 1200，分列 General 1000 / Unanswerable 200
RESULTS_DIR/MODEL/cross_old400.json      # 固定分母 400
RESULTS_DIR/MODEL/cross_hard400.json     # 固定分母 400
RESULTS_DIR/MODEL/reasoning_old100.json  # 固定分母 100
RESULTS_DIR/MODEL/reasoning_hard100.json # 固定分母 100
```

名称中的 `ordinary1190` 是原实验组件标识，评分分母为 1200。文件采用原脚本的
`paper -> QA -> item` 结构，记录分别保留金标 `answer` / `evidence_pages` 和预测
`answer_pre` / `evidence_pages_pre`（以及已有 raw 字段）。不得把模型预测当金标。
总分母始终为 2200，主指标是 Answer Accuracy、逐题宏平均 E-Precision/E-Recall/E-F1
和唯一预测页数的平均值 A-Pages。

一个结果记录的最小结构如下；其中 `answer` 和 `evidence_pages` 来自相应版本的
人工参考标注，`answer_pre` 和 `evidence_pages_pre` 来自模型预测，不能互相覆盖：

```json
{
  "paper_id": {
    "QA": {
      "qa_id": {
        "question": "The question from the frozen experiment input",
        "answer": "The reference answer",
        "evidence_pages": [2],
        "answer_pre": "The model's answer",
        "evidence_pages_pre": [2]
      }
    }
  }
}
```

这是结构说明，不是完整基准输入。每个模型必须提供全部五个组件文件，缺失预测仍留在固定
分母内。当前 CLI 的 `--model` 接受已注册的 11 个基线目录标识，以 `--help` 为准；
不应把任意新模型的结果冒充某个已注册模型来比较论文数值。

正式四文件 QA、PDF 和论文文本未修改。当前四文件与历史结果自带金标存在版本差异，
因此不能将两个版本混合后声称复现原实验。旧的扁平 `--predictions` CLI 已替换为上面的
五组件输入；`scripts/validate_submission.py` 保留为当前模型输出格式的诊断工具，
其严格格式状态不再是 v4 评分门禁。

## 原实验缓存回放（无需 GPU）

```bash
python evaluation/evaluate.py \
  --results-dir data/results \
  --offline \
  --judge-cache sxz/evaluation_11models_5datasets_qwen36_calibrated_fixeddenom_v4/qwen36_answer_judge_cache.json \
  --subject-xlsx sxz/final_2200_classification_statistics.xlsx \
  --reference-summary data/results/evaluations/summary_11models_77rows_calibrated_fixed_denominator.csv \
  --output-dir data/results/sxz_v4_replay_new
```

默认执行全部 11 模型，使用 `--model Qwen3-VL-8B` 可只执行一个模型。
回放会核验每个缓存条目的提示词、实际 Judge 输入、键及原始回复与标签的一致性；
存在任何所需缓存缺失即在评分前失败。`sxz/` 缓存只读，不复制为可继续生成的新运行缓存。
历史文件不在干净 Git clone 中，需要单独取得；评分代码运行不依赖导入 `sxz/`。

输出包括每个模型的 `details.csv` / `summary.csv`、全体 `summary.csv`、
`paper_comparison.csv`、`run_binding.json` 和 `report.json`。证据指标在内部汇总 CSV 中
为 0–1，Table 2/report 中的百分比为 0–100，A-Pages 不乘 100。

`--subject-xlsx` 用原始独立学科清单连接逐题结果；报告会记录集合外记录与集合内缺失项。
不提供该参数时只生成 Table 2。`--reference-summary` 是可选的历史数值对照，不参与评分。

`evaluate.py` 成功完成评分返回 0；历史汇总数值不一致返回 2；输入或 Judge 失败返回 1。
`reproduce.py` 使用完全相同的参数和评分器，但只要已计算的论文单元格有差异就返回 2。
因此带学科清单的全量 `reproduce.py` 比较会返回 2 并报告 Claude 的已知差异；
同一组输入的 `evaluate.py` 可正常返回 0，因为评分本身已完成。未提供学科清单时，
`table3_display_matches: 0` 表示未计算 Table 3，不是该表全部不匹配。

## 新本地 Judge 运行

```bash
python evaluation/evaluate.py \
  --results-dir data/results --model Qwen3-VL-8B \
  --gpu 2 --model-dir models/Qwen3.6-27B \
  --output-dir data/results/sxz_v4_new_judge_run
```

需要内部 Python 3.12+ ML 环境、可用 A800 和完整本地权重；`requirements-eval.txt`
只覆盖 CPU 评分与测试。ML 环境安装见 [维护上手指南](../docs/current/GETTING_STARTED.md#2-准备环境与资产)。
此命令仅重新判分已有预测，不执行 PDF 答案推理。使用原脚本的加载及生成函数：seed=42、
BF16、greedy、32 个新 token、禁用 thinking、原始 chat template 调用和解码设置。
记录当前权重文件哈希及 PyTorch/Transformers 版本；使用独立 `judge_cache.json`。
输入、代码或权重变化必须换新输出目录。禁止把原实验缓存混入新生成运行。

**同规则不保证重新生成的标签逐条相同。** 原实验没有保存完整权重 revision/字节绑定；
原始缓存回放能够精确复算已记录实验，重新调用当前权重则是新的 Judge 运行，必须实测。
本次验证未启动新的 GPU 判分。

## 迁移与测试

此前二分类/current-release 分差诊断的报告保留为历史证据；相关旧命令已停用，避免在
新评分器上错误续跑旧缓存。包括 `check_reproduction.py`、`investigate_gaps.py`、
`audit_score_provenance.py` 及旧本地 smoke、rescore、PDF-to-score、配对判分诊断脚本。
当前命令以本文件为准。内部 `src/pku_qa/evaluation/` 的二分类诊断继续固定读取
`prompts/paper_semantic_judge.txt`；它不生成此处的 v4 论文评分。

```bash
python -m pytest -q tests/test_sxz_evaluation.py tests/test_judge.py tests/test_e2e.py tests/test_metrics.py
python .agents/skills/pku-qa-maintainer/scripts/sync_project_docs.py --sync
python -m pytest -q
```

回归测试覆盖原脚本函数哈希与逐题对照、完整提示词、恢复优先级、无金标泄漏、PARTIAL
零分、技术失败的独立证据分、固定分母、缓存篡改和缺失、学科集合、只读路径与符号链接。
