# ScienceDoc：多模态科学论文问答基准

当前公开评测入口、安装与验证命令见 [evaluation/README.md](evaluation/README.md)。本次工作不修改任何 QA JSON；历史复现差异见 [审计报告](docs/reports/official_evaluation/FINAL_REPORT.md)。

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

### 1. 获取项目并检查发布数据

需要 Python 3.12 或更新版本。以下步骤不需要 GPU、PDF、API 密钥或第三方 Python 包：

```bash
git clone https://github.com/barcelonaChinesegit/SciDoc.git
cd SciDoc
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200
```

成功时输出 `"status": "valid"`、`"qa_count": 2200`、四组件数量和 `"evaluation_pdfs": 712`。该命令只读四个正式 JSON，核验共享字段、全局 ID、数量和 manifest 哈希，并重算模态及文档组合统计。`pdf_check: "not_requested"` 表示此步尚未检查外部 PDF。

读取一道真实题目：

```python
import json
from pathlib import Path

data = json.loads(Path("data/qa/7.final_2200/ordinary_qa.json").read_text())
paper_id, paper = next(iter(data.items()))
qa_id, qa = next(iter(paper["QA"].items()))
print(paper_id, qa_id)
print(qa["question"])
print(qa["answer"])
print(qa["evidence_pages"])
```

### 2. 准备 PDF 并检查证据页

PDF 是单独管理的外部资产，Git 克隆不含 PDF 和模型权重。向项目维护者取得与 [PDF 资产清单](data/pdf_assets_manifest.json) 匹配的文件，平铺到 `data/pdfs/`。请按清单使用 `paper_*`、`source_*`、`z_cross_*` 文件名；重新下载的另一版本不能替代被冻结的 PDF。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pypdf==6.10.2
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200 --check-pdfs
```

此步核验最终 712 份 PDF 的 SHA-256、可读性和全部金证据页范围。成功时 `pdf_check` 为 `hashes_readability_and_evidence_bounds_valid`。

### 3. 内部历史实验运行器（不代表论文分数复现）

完整推理需要 CUDA 环境和本地模型。安装及显存要求见 [上手指南](docs/current/GETTING_STARTED.md)。准备 `models/Qwen3-VL-4B-Instruct/`、`models/Qwen3-VL-8B-Instruct/` 和 `models/Qwen3.6-27B/` 后，先打印计划：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
PYTHONPATH=src python -m pku_qa.workflows.reporting.run_final_2200_evaluation \
  --print-plan -- --dynamic-a800 --a800-gpus 2 3 4 5
```

GPU 编号必须按本机情况调整。确认资源和输入后，移除 `--print-plan` 执行；可用 `--component reasoning` 仅运行一个组件。默认依次运行 ordinary、unanswerable、reasoning、cross_pdf，各自执行 4B → 8B → Judge → 严格报告，结果写入 `data/results/evaluations/final_2200/<component>/`。计划打印不会启动推理或调用付费 API。

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

论文主指标为语义 Judge Answer Accuracy、逐题宏平均 E-Precision / E-Recall / E-F1 和 A-Pages，固定分母为 2,200。完全页集合匹配和联合正确性仅是审计诊断，不是主指标。`evaluation/` 按论文实现，不使用字符串、别名或数值预匹配；`src/pku_qa/evaluation/` 的旧规则优先评分已删除，Judge 统一使用论文原版提示词；内部流程仍不构成论文结果复现。历史 Table 2 的 99 个两位小数单元格可由 v4 记录重聚合，但旧输出和当前金标存在版本差异，不能声称新评测器已经复现论文。

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
| `src/pku_qa/evaluation/` | 协议、推理、Judge、报告和断点恢复 |
| `src/pku_qa/workflows/` | 生成、清洗、复审、选择、统计和运维入口 |
| `tools/internal/experiment_console/web/`、`src/pku_qa/services/` | Web 控制台与后端 |
| `tests/`、`schemas/` | 自动测试和数据结构约束 |
| `scripts/dataset_construction/` | 早期论文获取、基础 QA Notebook 与提示词 |
| `data/archive/` | 历史实验与构建记录；Git 中只提供说明和清单 |

PDF 与模型遵循各自来源许可；仓库不包含个人账户数据库、会话或 API 密钥。`sxz/` 属于其他协作者，只可读取参考。
