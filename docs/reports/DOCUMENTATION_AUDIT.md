# README 与全库描述核对（2026-09-13）

> 历史记录：文中旧路径与当时测试结果保留用于追溯。当前 Web 源码在
> `tools/internal/experiment_console/web/`；当前论文协议和复现结论见
> [现行 sxz v4 评分验证](official_evaluation/SXZ_V4_ALIGNMENT.md)。

本次按最终四文件核对项目描述，共逐文件扫描 284 份可维护文本和结构化文件，
包含 152 份 Python、26 份 Markdown、6 份 Notebook。
[扫描明细](documentation_audit_files.csv) 列出路径和范围。扫描覆盖源码、配置、文档、
Notebook 单元、QA/审计 JSON 与归档清单，命中后按具体代码和数据复核；这不表示对每段
历史代码重新运行实验或对所有论文重新作语义人工审核。

另读取三份 DOCX 的段落、表格、关系与内嵌图，核对四份最终 QA、发布 manifest 和统计
工作簿，并验证最终使用的 712 份 PDF 的哈希、可读性和证据页范围。外部模型、依赖、
构建缓存、运行账户与日志不属于文案维护对象；`sxz/` 保持只读且未写入。历史压缩包和
审计清单记录过去版本，不通过全局替换改变历史证据。

| 问题 | 修正 |
| --- | --- |
| README 缺少可在新克隆运行的入口 | 增加标准库只读检查、读取样例、PDF 部署检查和正式评测 Quick Start |
| 旧 4,211/100/400 被标成当前主评测 | 文档和注册器统一说明最终四文件、历史来源与分类镜像 |
| 文档仍用迁移前目录 | 修正 review 服务/运行目录、QA 编号目录与第一阶段 Notebook 新位置 |
| 人工审核手册的七张图片不存在 | 换成 docs/assets 下三张当前界面 PNG，采用相对链接 |
| 登录页和首页写五个正式文件 | 改为四个，并在 Web 测试中约束 |
| 网页指南使用 figure/equation 模态名 | 改为 text、image、table、formula |
| 草稿写 734 PDF、260 合并文件 | 改为 712、238；单论文 474 |
| 草稿模态组合为 1306/333/290 | 按最终数据改为纯文本 982、文本＋公式 609、文本＋表格 189 |
| 源论文领域数字无法由四文件直接复算 | 改用 2,200 题领域统计，图表生成器读取最终 JSON，明确分母 |
| 历史分数被当成当前发布结论 | 历史 400 报告、论文诊断图与附录表明确版本、分母及绑定核验边界 |
| 附录命令/审核枚举过时 | 修正 expected_qa_sha256、expected_pdf_corpus_sha256、KEEP/FIX/REJECT |
| 论文可能引用内部维护页面 | 检查全部 DOCX XML/关系，无 CURRENT_STATUS 或 PROJECT_INVENTORY 引用；README 移除两项导航 |

最终四个 QA 的内容和 manifest 哈希保持原样。本次没有重新生成题目或运行模型评测。
三份 DOCX 与论文插图保存在本机 `论文/写作材料/`，该目录由现有 Git 忽略规则排除，
不会随普通 Git 提交自动上传。英文参考稿的统计图与中文 Dataset 草稿同步更新。

验证入口：

```bash
PYTHONPATH=src python -S -m pku_qa.workflows.operations.inspect_final_2200
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200 --check-pdfs
PYTHONPATH=src python -m pku_qa.workflows.reporting.build_final_2200_classification --check-only
PYTHONPATH=src python -m pku_qa.workflows.reporting.run_final_2200_evaluation --print-plan
python -m pytest -q
(cd task_queue_web && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy npm test)
PYTHONPATH=src python -m pku_qa.workflows.operations.check_web_stack
```

首次检查另在只复制 src 和最终五个 JSON、关闭 site-packages 的临时目录验证，
确认不依赖 PDF、上游基线或分类镜像。依赖文件使用 pip --dry-run 完成解析。
Web 变更已构建并重启应用与数据 API；公网检查要求本地三端点、隧道后端和公网均为 200。
静态文档图片已逐张打开检查；新增回归检查会拒绝指向不存在本地文件的 README/当前文档链接。
