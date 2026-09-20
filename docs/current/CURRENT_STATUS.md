# 当前项目状态

## 2026-09-20 论文协议审计更新

公开 submission 的验证、计分和命令以 [evaluation/README.md](../../evaluation/README.md) 为准。
论文主指标是语义 Answer Accuracy、逐题宏平均 E-Precision / E-Recall / E-F1 和 A-Pages；
精确页集合匹配与联合正确性只是审计诊断。`src/pku_qa/evaluation/` 的规则优先匹配与
内部报告属于历史实验实现，不能据此宣称复现当前论文。历史 v4 重聚合匹配 Table 2 的
99/99 个显示值、Table 3 的 98/99 个显示值；新二分类 Judge 的正式历史运行配置仍未恢复。
差异及完整证据见 [official evaluation 审计](../reports/official_evaluation/FINAL_REPORT.md)。
本次整理只读验证 QA；问题、答案、证据、元数据和 manifest 均不修改。


核对日期：2026-09-20。本文件用于维护；论文统计引用最终发布文件及分类工作簿。

## 当前发布

| 文件 | QA | PDF 记录 |
| --- | ---: | ---: |
| `data/qa/7.final_2200/ordinary_qa.json` | 1,000 | 398 |
| `data/qa/7.final_2200/unanswerable_qa.json` | 200 | 132 |
| `data/qa/7.final_2200/reasoning_qa.json` | 200 | 87 |
| `data/qa/7.final_2200/cross_pdf_qa.json` | 800 | 238 |

全局 ID 为 `QA0001`–`QA2200`，共 2,200 道简答题。组件间 PDF 有重叠，最终
共使用 712 份不同评测 PDF：474 份单论文、238 份合并 PDF。Cross-PDF 包含两文档题
621 道和三文档题 179 道。它们与完整外部资产目录的文件总数不是同一统计口径。

四文件与同目录 `rel__collection__final_2200__manifest.json` 是当前唯一发布输入。
Reasoning 两批 100、Cross-PDF 两批 400 及 4,211 题基线是历史构建来源。
Web 与审核 SQLite 使用最终四文件及全局身份；审核运行数据在 `data/web/review/`。
两条人工删除旧 Cross-PDF 题及补题身份可从最终 QA provenance、来源选择账本和审核记录
追溯；当前 schema_version 2 的 manifest 只保存四组件，不含旧 `post_review_supplements` 字段。

## 当前模态统计

最终四文件的 2,200 条均保留 PDF 页图模态复核记录。模态出现数允许一题计入多个列：

| 文件 | QA | 文本 | 图像 | 表格 | 公式 | 单模态 | 多模态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | 1,000 | 959 | 97 | 79 | 256 | 615 | 385 |
| unanswerable | 200 | 169 | 9 | 37 | 38 | 147 | 53 |
| reasoning | 200 | 200 | 29 | 33 | 56 | 89 | 111 |
| cross_pdf | 800 | 800 | 172 | 162 | 416 | 202 | 598 |
| 总计 | 2,200 | 2,128 | 307 | 311 | 766 | 1,053 | 1,147 |

精确组合中，纯文本 982、文本＋公式 609、文本＋图像 192、文本＋表格 189。
工作簿 `data/qa/7.final_2200/final_2200_classification_statistics.xlsx` 按四组件汇总，
包含八个一级领域、21 个问题小类及逐题明细。

## 核验与运行

```bash
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200 --check-pdfs
PYTHONPATH=src python -m pku_qa.workflows.reporting.run_final_2200_evaluation --print-plan
```

manifest 结构核验不检查 PDF 可读性；第二条命令补充 PDF 哈希与证据页范围核验。
正式运行从 manifest 锁定四组件，结果写入 `data/results/evaluations/final_2200/`。
单 PDF Full/Oracle 专项由最终普通与不可回答题组成 1,200 题；闭卷使用普通 1,000 题，
逐页消融数量按实际证据生成。旧 `Challenge119` 是 1,190 题旧输入的抽样结果。

## 后续维护

每次金标准变更后重建 manifest、分类镜像和工作簿，并重新冻结评测输入。
旧模型分数、历史抽样与新四文件结果须分别报告，不能由已有诊断记录推断当前整集性能。
历史材料在 `data/archive/final_2200_cleanup_20260911/` 按用途保留，禁止覆盖式恢复到
当前规范路径。`sxz/` 严格只读。
