# ScienceDoc：多模态科学论文问答基准

当前公开评测入口、安装与验证命令见 [evaluation/README.md](evaluation/README.md)。默认评分已切换为 sxz v4 原实验规则；QA JSON 不修改，验证见 [对齐报告](docs/reports/official_evaluation/SXZ_V4_ALIGNMENT.md)。

**简体中文** | [English](README.md)

SciDoc 面向科学论文 PDF 的阅读理解、证据定位和跨论文推理。问题覆盖正文、图像、表格与公式，要求模型给出简短答案及其 PDF 物理证据页；对论文无法支持的问题，模型应准确拒答。

项目提供 **2,200 条人工终审简答 QA**、数据校验工具、可恢复的推理与判分流水线，以及多人审核 Web 控制台。数据制作方法与论文 Dataset、附录中的问题构建、独立复审、人工核验和证据评估相对应。

## 数据集

当前发布输入只有 [`data/qa/7.final_2200/`](data/qa/7.final_2200/) 下的四个 QA 文件：

| 文件 | QA 数 | PDF 数¹ | 任务 |
| --- | ---: | ---: | --- |
| [ordinary_qa.json](data/qa/7.final_2200/ordinary_qa.json) | 1,000 | 398 | 单论文事实理解、比较和计算 |
| [unanswerable_qa.json](data/qa/7.final_2200/unanswerable_qa.json) | 200 | 132 | 识别论文证据不足，准确拒答 |
| [reasoning_qa.json](data/qa/7.final_2200/reasoning_qa.json) | 200 | 87 | 联合多个必要事实进行单论文多步推理 |
| [cross_pdf_qa.json](data/qa/7.final_2200/cross_pdf_qa.json) | 800 | 238 | 联合两篇或三篇论文的信息回答 |

¹ PDF 数是各组件中的文件数，组件间存在重复。最终评测共使用 **712 份不同 PDF：474 份单论文 PDF、238 份合并 PDF**。合并 PDF 不是新增的原始论文。Cross-PDF 中 621 题依赖两篇论文，179 题依赖三篇论文。

所有题目使用全局 ID `QA0001`–`QA2200`，Cross-PDF 占最后 800 个 ID。最终数据覆盖八个一级领域、21 个问题小类；单模态题 1,053 道，多模态题 1,147 道。模态组合中，纯文本 982 道，文本＋公式 609 道，文本＋图像 192 道，文本＋表格 189 道，其余为其他组合。

[发布 manifest](data/qa/7.final_2200/rel__collection__final_2200__manifest.json) 固定四文件的数量、身份与内容哈希；[分类统计工作簿](data/qa/7.final_2200/final_2200_classification_statistics.xlsx) 提供分组件统计和逐题明细。上游 6,204 条原始 QA、4,211 条清洗基线及分批源组件属于构建谱系，不与最终 2,200 条相加。

## Quick Start

### 1. 安装并检查评测包

独立 `evaluation/` 使用 Python 3.10+。从新克隆开始：

```bash
git clone https://github.com/barcelonaChinesegit/SciDoc.git
cd SciDoc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python evaluation/evaluate.py --help
python -m pytest -q tests/test_sxz_evaluation.py tests/test_judge.py tests/test_e2e.py tests/test_metrics.py
```

这些检查不需要 GPU、PDF、模型、API 密钥或历史结果。干净克隆没有协作目录 `sxz/`
时，只跳过与本机原脚本直接比较的测试；随包提供的函数哈希与评分测试仍会执行。

### 2. 回放一个模型的原实验评分

向维护者取得以下原始实验材料，Git 不包含这些文件：

| 材料 | 本机约定位置 |
| --- | --- |
| Qwen3-VL-8B 的五个结果 JSON，保留金标及预测字段 | `data/results/Qwen3-VL-8B/` |
| 原始 v4 Judge 缓存 | `sxz/evaluation_11models_5datasets_qwen36_calibrated_fixeddenom_v4/qwen36_answer_judge_cache.json` |

五个文件的名称与结构见 [评测输入说明](evaluation/README.md#输入与安装)。
下面使用现有工作区路径，也可用 `--results-dir`、`--judge-cache` 指向其他位置的副本；
评分器不导入或写入 `sxz/`。

```bash
python evaluation/evaluate.py \
  --results-dir data/results --model Qwen3-VL-8B --offline \
  --judge-cache sxz/evaluation_11models_5datasets_qwen36_calibrated_fixeddenom_v4/qwen36_answer_judge_cache.json \
  --output-dir data/results/quick_start_qwen8b
```

该命令不需要 GPU、PDF 或网络：从结果文件恢复预测，匹配原始缓存判定，重新计算指标。
使用原始材料时，预期 `table2_display_matches: 9`、`paper_differences: []`，退出码 **0**。

### 3. 查看结果

```bash
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("data/results/quick_start_qwen8b/report.json").read_text())
print(report["execution"])
print(report["table2"]["Qwen3-VL-8B"])
PY
```

运行类型为 `recorded_cache_replay`；显示到两位小数时，All 为 **68.32%**、E-F1 为
**40.66%**、A-Pages 为 **8.44**。逐题明细在
`data/results/quick_start_qwen8b/Qwen3-VL-8B/details.csv`，输入、代码及缓存绑定在
`run_binding.json`。原命令可再次执行以核对同一运行；输入或代码改变时使用新输出目录。

移除 `--model` 可评测全部 11 个模型；提供原始 `--subject-xlsx` 可生成 Table 3，
提供 `--reference-summary` 可逐项对照历史汇总。见
[完整回放命令](evaluation/README.md#原实验缓存回放无需-gpu)。

### 4. 新运行 Judge（可选，需要 GPU）

安装内部 Python 3.12+ ML 环境、准备空闲 A800 和本地 Qwen3.6-27B 权重后执行：

```bash
python evaluation/evaluate.py \
  --results-dir data/results --model Qwen3-VL-8B \
  --gpu 2 --model-dir models/Qwen3.6-27B \
  --output-dir data/results/quick_start_qwen8b_new_judge
```

按本机情况选择 GPU 编号。该命令用 v4 提示词和生成配置重新判断已有模型答案，
不会重新读取 PDF 生成答案；新运行不传 `--judge-cache`。
依赖、断点与权重身份限制见 [新 Judge 运行说明](evaluation/README.md#新本地-judge-运行)。
完整 PDF 推理与内部二分类诊断另见 [维护上手指南](docs/current/GETTING_STARTED.md)。

## 当前数据与格式检查

若只检查 Git 中的四个发布 JSON，使用 Python 3.12+，无需第三方依赖：

```bash
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200
```

成功时输出 `status: valid`、`qa_count: 2200` 和 `evaluation_pdfs: 712`。
取得 [清单](data/pdf_assets_manifest.json) 中的 PDF 并平铺到 `data/pdfs/` 后，可检查
完整的哈希、可读性和金证据页范围：

```bash
python evaluation/preflight.py --output data/results/preflight.json
```

`scripts/validate_submission.py --predictions predictions.jsonl` 检查添加 `qa_id` 的
扁平模型输出格式，需要发布集 PDF。自带 `examples/predictions.example.jsonl` 只有两题，
会因缺失其余题目返回 1。这是格式诊断，不是 v4 评分入口；v4 使用前述五个自带金标的
结果文件。当前四文件与原实验金标有版本差异，原实验回放不能混用当前金标。

## 构建方法

基础构建从 arXiv 论文检索、分类整理和 PDF 检查开始，得到 703 篇论文的 6,204 条原始 QA；格式清洗、去重、简答化和证据修正后形成 693 篇论文的 4,211 条单论文基线。随后构造主题相关但证据不足的不可回答题、多事实 Reasoning 题，以及多文档 Cross-PDF 题。

生成流程包括基于已核验 QA 事实的组合生成和全文 PDF 生成。程序检查字段、页码、重复、答案泄漏及必要依赖；专项候选经过 Claude Sonnet 5 与 Gemini 2.5 Flash 独立复审和证据恢复，再经人工逐题核验。最终四文件的全部 2,200 条另完成了基于 PDF 页图的模态复核。生成、审核和历史诊断记录用于追溯，不能替代当前独立金标准。

## 数据与评测约定

JSON 顶层为 `{paper_id: paper}`，每个 paper 内的 `QA` 为 `{qa_id: qa}`。六个共享必填字段为 `question`、`answer`、`evidence_pages`、`modal_types`、`question_type`、`question_category`，完整定义见 [最终 QA Schema](schemas/final_2200_qa.schema.json)。最终题目没有选择题 `options`；专项来源和审核字段保留在各 QA 中。

| 协议 | 输入 | 模型输出 |
| --- | --- | --- |
| `question_only` | 仅问题；闭卷对照与难度检查 | 答案纯文本 |
| `pdf` | 问题和带 `[Page N]` 标签的页面图像 | `{"answer_pre":"…","evidence_pages":[1,3]}` |

Full、Oracle、证据页消融仅改变 `pdf` 的页面选择。所有页码从 1 开始，采用 PDF 文件物理页；Cross-PDF 使用合并文件页码。拒答标签精确为 `Unanswerable`；PDF 模式必须返回 `{"answer_pre":"Unanswerable","evidence_pages":[]}`。

论文主指标为语义 Judge Answer Accuracy、逐题宏平均 E-Precision / E-Recall / E-F1 和 A-Pages，固定分母为 2,200。`evaluation/` 使用 sxz v4 原始三分类提示词、预测恢复、结果自带历史金标及证据计分规则；只有 CORRECT 计正确。55 文件与原始缓存重算匹配历史汇总全部 1,540 个数值、Table 2 的 99/99 和 Table 3 的 98/99 个显示值；Claude 学科表总分差异保留。评分实现和缓存回放已验证；新 GPU Judge 运行不保证逐条复现旧标签。论文当前二分类提示词与采用的实验规则仍有文字差异，本次未修改论文。当前命令以 [evaluation/README.md](evaluation/README.md) 为准；内部二分类诊断不作为 v4 评分入口。

## 人工审核与文档

[打开 Web 控制台](https://pku.chenzijian.com/)。使用个人应用账户登录，管理员按稳定 QA 身份分配范围；审核者对照 PDF 核验问题、答案和证据，修改与删除保留快照及审计记录。公网入口经过 VPS HTTPS 反向隧道连接学校内网服务。

- [项目架构](docs/current/ARCHITECTURE.md)：协议、模块、目录和字段约束。
- [上手指南](docs/current/GETTING_STARTED.md)：安装、预检、评测和排查命令。
- [人工审核手册](docs/current/MANUAL_REVIEW_GUIDE.md)：逐题标准、界面说明与审核流程。
- [Web 控制台](docs/current/WEB_CONSOLE.md)：账户、分配、权限和部署。
- [QA 数据集目录](docs/current/DATASET_CATALOG.md)：最终输入、上游来源和过程材料的用途及权限。

## 项目目录

| 路径 | 用途 |
| --- | --- |
| `data/qa/7.final_2200/` | 四文件发布集、manifest、分类工作簿 |
| `data/qa/1.base/` 至 `6.review/` | 基础题源、不可回答镜像、Reasoning/Cross-PDF 源组件、人工审核来源与复核证据 |
| `evaluation/` | sxz v4 评分、原实验缓存回放、当前数据与格式检查 |
| `src/pku_qa/evaluation/` | 内部推理、二分类诊断、报告和断点恢复 |
| `src/pku_qa/workflows/` | 生成、清洗、复审、选择、统计和运维入口 |
| `tools/internal/experiment_console/web/`、`src/pku_qa/services/` | Web 控制台与后端 |
| `tests/`、`schemas/` | 自动测试和数据结构约束 |
| `scripts/dataset_construction/` | 早期论文获取、基础 QA Notebook 与提示词 |
| `data/archive/` | 历史实验与构建记录；Git 中只提供说明和清单 |

PDF 与模型遵循各自来源许可；仓库不包含个人账户数据库、会话或 API 密钥。`sxz/` 属于其他协作者，只可读取参考。
