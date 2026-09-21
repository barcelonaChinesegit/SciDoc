# 报告与历史诊断

现行评分实现和全量验证见 [sxz v4 对齐报告](official_evaluation/SXZ_V4_ALIGNMENT.md)。
安装、输入与运行命令见 [evaluation Quick Start](../../evaluation/README.md#quick-start)。

| 文件 | 范围 | 使用边界 |
| --- | --- | --- |
| [sxz v4 对齐验证](official_evaluation/SXZ_V4_ALIGNMENT.md) | 当前评分规则与 55 文件原始缓存回放 | 历史汇总 1,540/1,540、Table 2 99/99、Table 3 98/99；未重新调用 GPU Judge |
| [实验来源核对](official_evaluation/PROVENANCE_RECONCILIATION.md) | 2026-09-20 输入版本、旧缓存与新二分类分差归因 | 历史诊断；不替代当前 v4 评分 |
| [原评测审计](official_evaluation/FINAL_REPORT.md) | 2026-09-20 二分类实现及历史实验调查 | 旧接口和诊断命令已停用，记录保留追溯 |
| [文档入口更新](EVALUATION_DOCS_UPDATE_20260921.md) | 当前 Markdown 与 Quick Start 核验 | 说明文档覆盖范围及命令验证 |
| [Cross-PDF 证据页错误分析](CROSS_PDF_EVIDENCE_AUDIT.md) | 旧第一批 400 题及其中 100 题人工抽样 | 历史诊断，不是当前 800 条 Cross-PDF 或 2,200 条整集结果 |
| [2026-09-13 文档核对](DOCUMENTATION_AUDIT.md) | 当时的数据、目录与说明核对 | 旧路径及测试数保留，当前入口以本页为准 |

当前发布数据见 [数据说明](../../data/qa/README.md)。原实验评分使用结果文件内冻结的
参考标注；不要用当前四文件覆盖后声称复现同一次实验。论文主要证据指标为逐题宏平均
PRF 和 A-Pages，精确页集合匹配为诊断；历史 ±1 页分析不改变当前评分规则。

Claude Table 3 All 的 1.27 个百分点差异是学科集合与 Table 2 汇总集合不同；
同规则缓存回放保留该差异。新 Judge 生成与原缓存回放应分别描述。
