# 项目架构、目录与字段说明

## 2026-09-20 论文协议审计更新

公开 submission 的验证、计分和命令以 [evaluation/README.md](../../evaluation/README.md) 为准。
论文主指标是语义 Answer Accuracy、逐题宏平均 E-Precision / E-Recall / E-F1 和 A-Pages；
精确页集合匹配与联合正确性只是审计诊断。`src/pku_qa/evaluation/` 的规则优先匹配与
内部报告属于历史实验实现，不能据此宣称复现当前论文。历史 v4 重聚合匹配 Table 2 的
99/99 个显示值、Table 3 的 98/99 个显示值；新二分类 Judge 的正式历史运行配置仍未恢复。
差异及完整证据见 [official evaluation 审计](../reports/official_evaluation/FINAL_REPORT.md)。
本次整理只读验证 QA；问题、答案、证据、元数据和 manifest 均不修改。


## 1. 系统边界

本项目负责本仓库中除 `sxz/` 外的 QA 数据审计、模型推理、统一判分、报告、
断点恢复和任务监控。`sxz/` 属于其他同学，作为只读参考目录保留；项目工具不在其中
写入、生成缓存或修改文件。

正式评测的不可变原则：

1. 被测模型协议只有 `question_only` 与 `pdf` 两种。
2. Full、Oracle、消融仅改变 PDF 页选择，不产生第三种协议。
3. `Unanswerable` 大小写与拼写必须完全一致；PDF 拒答页为 `[]`。
4. 4B、8B 使用同一 QA、同一 PDF 集、同一提示协议和同一 Judge 逻辑。
5. 金标准永远来自独立 `--qa-json`，不得从模型输出反推。
6. 论文报告只接受完整指纹与逐题绑定；没有“旧产物兼容/宽松报告”开关。

## 2. 内部历史实验数据流

```text
规范 QA JSON + PDF 文件
        │  数据协议校验、QA/PDF SHA-256
        ▼
run_inference.py ── 4B/8B 原始输出、规范化审计、协议指纹
        │  每篇论文原子检查点
        ▼
run_judge.py ── 规则判分；必要时调用 27B 语义 Judge
        │  inference_binding_sha256 + judge_protocol_fingerprint
        ▼
run_report.py ── 分母、键集合、原始输出、PDF 清单重新核验
        │
        ├─ Reasoning 专项三指标报告
        └─ 最终单 PDF 四变体汇总与技术 HTML 报告
```

`src/pku_qa/evaluation/run_hard_eval.py` 编排单个数据集的 4B → 8B → Judge 4B → Judge 8B →
Report。`src/pku_qa/workflows/reporting/run_single_pdf_challenge_gpu_queue.py` 顺序编排最终单 PDF
基准的四个评测变体。外层 `task_queue_*` 负责整项实验依赖、进程恢复和日志，内层
`DurablePaperQueue` 负责逐论文断点；两层用途不同。

任务队列 supervisor 通过 `pku_qa.services.task_queue.task_queue_daemon` 包模块启动
daemon，并显式传递 `src` 包路径。可恢复任务重新启动时以本次启动时间建立新的心跳
窗口；磁盘上的旧 scheduler checkpoint 可用于恢复计数，但必须在配置的 stale 窗口内
刷新进程心跳，否则才判为真正卡死，不能在刚启动时因旧时间戳被误杀。

最终单 PDF 评测最后先生成 `metrics.json/csv`、Markdown 与 report artifact，再由
Data Analytics 报告渲染器生成独立 HTML。编排器显式解析 Node 与 npm CLI，并把
Node 所在目录加入子进程 `PATH`，因此 systemd 无交互环境不依赖登录 shell；前端
构建脚本会隔离 `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`，避免 Wrangler 误解析
非 HTTP 代理协议。

## 3. 两种协议

| 属性 | `question_only` | `pdf` |
| --- | --- | --- |
| 模型输入 | 只有问题文本 | 问题文本 + 带外部页号的页面图像 |
| 页面策略 | 只能是 `full`（实际不读取 PDF） | `full`、`oracle`、`qa_field` |
| 模型输出 | 原样答案文本 | 只允许规范 JSON |
| `require_structured_output` | `false` | `true` |
| 答案题证据要求 | 不适用 | 至少一个直接支持页 |
| 不可回答输出 | `Unanswerable` | `{"answer_pre":"Unanswerable","evidence_pages":[]}` |
| 内部精确页集合诊断 | 固定为可用，不参与主指标 | 与金标页集合完全相等 |

`src/pku_qa/evaluation/evaluation_protocol.py` 是这些规则、版本号、强制字段、哈希和队列 contract 的
唯一真源。CLI 只接收 `--input-mode`，证据要求由协议派生，不能由调用者覆盖。

PDF 原始 JSON 若仅有外层空白或页码顺序可确定性规范化，系统同时保存
`raw_model_output`、其 SHA-256、规范化后的 `model_output` 和
`deterministic_normalizations`。推理只做有限次格式纠错；仍然非法时逐字保存最后一次
模型输出及其哈希，让整批实验继续并由 Judge 计错。非法 JSON、越界页、拒答携带
证据页等均标为 `output_is_legal=false`，不能被推理程序或 Judge 悄悄修复。

## 4. 顶层模块

| 模块组 | 文件 | 职责 |
| --- | --- | --- |
| 协议与评分 | `src/pku_qa/evaluation/evaluation_protocol.py`, `src/pku_qa/evaluation/qa_scoring.py` | 输入/输出 contract、精确拒答、类型化字符串/数值匹配、哈希绑定 |
| 模型与 PDF | `src/pku_qa/evaluation/eval_framework.py`, `src/pku_qa/evaluation/run_inference.py` | provider、模型身份、PDF 渲染/页预算、提示、逐题推理 |
| Judge 与报告 | `src/pku_qa/evaluation/run_judge.py`, `src/pku_qa/evaluation/run_report.py`, `src/pku_qa/evaluation/calculate_evidence_weighted_accuracy.py` | 规则优先判分、27B 语义判分、严格汇总、证据加权指标 |
| 实验调度 | `src/pku_qa/evaluation/run_hard_eval.py`, `src/pku_qa/evaluation/run_eval_pipeline.py` | 动态或静态 GPU 的完整实验阶段编排 |
| 断点与 GPU | `src/pku_qa/evaluation/durable_work_queue.py`, `src/pku_qa/evaluation/adaptive_gpu_pool.py`, `src/pku_qa/evaluation/gpu_reservation.py`, `src/pku_qa/evaluation/progress_logging.py` | 原子检查点、worker 重启、A800 显存门禁、心跳 |
| 外层任务队列 | `src/pku_qa/services/task_queue/task_queue_manager.py`, `src/pku_qa/services/task_queue/task_queue_daemon.py`, `src/pku_qa/services/task_queue/task_queue_runner.py`, `src/pku_qa/services/task_queue/task_queue_cli.py`, `src/pku_qa/services/task_queue/task_queue_api.py`, `src/pku_qa/services/task_queue/task_queue_tui.py`, `src/pku_qa/services/task_queue/task_queue_supervisor.py` | 实验依赖、SQLite 状态、日志、重试、API/TUI/Web 服务 |
| 人工审核 | `src/pku_qa/services/review/data_review_manager.py`, `src/pku_qa/services/review/data_manager_api.py` | QA 搜索与序号跳转、审核状态、可逆修改、稳定 QA 分配、用户/会话、角色授权与统一审计 |
| 分类工作流 | `src/pku_qa/workflows/{cleaning,generation,review,selection,reporting,operations}/` | 清洗、生成、复审、实验报告和运维入口；逐文件用途在自动清单中 |
| 测试 | `tests/{unit,integration,system}/` | 纯函数、跨模块流水线、服务与文档新鲜度 |

完整到单文件的用途、行数和当前 SHA-256 见 `docs/current/PROJECT_INVENTORY.md`。

人工校验网页的数据集注册器会递归扫描 `data/qa/**/*.json`，只纳入顶层记录含非空
`QA` 映射的文件；因此嵌套目录中的候选、历史扩展和仍带 QA 结构的过程材料不会被遗漏。
完整的当前条目、来源/获得方式、用途、统计和读写权限见自动生成的
`docs/current/DATASET_CATALOG.md`。正式数据保留可逆人工修改；模型输出、审核快照、断点、汇总等
过程条目在网页中可查看但明确标为只读，后端也会拒绝修改。

审核分配保存在 `review_assignments` 与稳定成员表中。管理员可通过 `/api/assignments` 管理
全部范围；普通登录用户只能通过 `/api/assignments?scope=mine` 读取自己的分配及其审核进度。
Web `/data` 页和 `/profile` 页都用这个只读接口展示 JSON、1-based 起止序号、题数和已完成统计，
因此管理员配置后无需额外通知，用户登录即可确认工作范围；权限校验仍由逐题写接口执行。
管理员页面另通过管理员专用的 `/api/reviewer-progress` 查看每个 reviewer 账户的聚合进度，
包括无分配账户、已完成/已分配、保留/修改/删除和最近完成时间；聚合基于稳定成员表，不会因
JSON 前序题删除或序号变化而把进度转移给其他人。

论文审核界面插图由 `论文/写作材料/图/图表生成代码/capture_manual_review.py`
从已构建的英文 `/data` 页重新截图，使用浏览器内只读 release fixture，保留
QA1816 和真实物理 PDF 页 16/57，不写账户、审核事件或正式数据。数据集标题、
按钮及元数据支持换行；PDF 默认收起缩略图侧栏。截图检查中文与水平溢出，
按实际面板尺寸生成紧凑画布，再同步至 `论文/ICLR2027_ScienceDoc/figures/`。

### 4.1 JSON 文件命名

数据集级文件名使用 `<status>__<family>__<purpose>__<batchNN>__n<count>[__vN].json`。
`rel__` 是可供当前评测或发表前校验的正式数据，`work__` 是过程性材料。批次从
`batch01` 起，跨 PDF 两批分别为 `batch01__n400` 和 `batch02__n400`。逐论文 checkpoint、
模型原始输出和运行态 manifest 使用稳定 ID 或固定 `manifest.json`，其用途由目录和 README
说明，不能误当成独立 QA 输入。

## 5. 目录说明

| 目录 | 当前用途 | 是否生成 |
| --- | --- | --- |
| `docs/current/` | 当前架构、状态、上手与人工审核说明 | 清单和数据集目录为生成 |
| `docs/reports/` | 仍有效的专项报告与索引 | 否 |
| `schemas/` | Single-PDF QA JSON Schema | 否 |
| `src/pku_qa/evaluation/` | 统一协议、推理、Judge、报告、GPU 与断点引擎 | 否 |
| `src/pku_qa/services/` | 任务队列和人工审核后端 | 否 |
| `src/pku_qa/workflows/` | 按 cleaning/generation/review/selection/reporting/operations 分类的入口 | 否 |
| `tests/{unit,integration,system}/` | 三层 Python 自动测试 | 否 |
| `scripts/dataset_construction/` | 早期论文获取与基础 QA Notebook、辅助下载器和提示词 | 否 |
| `tools/internal/experiment_console/web/` | 本地 Next/vinext 任务与数据审核控制台 | 构建产物生成 |
| `deploy/systemd/` | 本地 Web/API/daemon 健康守护的可复现 user units | 否 |
| `data/qa/1.base/` | 上游基础数据及最终普通 1000 的只读分类镜像 | 否 |
| `data/qa/2.unanswerable/` | 最终不可回答 200 的只读分类镜像 | 否 |
| `data/qa/5.human_reviewed/` | 旧单 PDF 数据同步仍使用的人工审核 authority 983 | 否 |
| `data/qa/3.reasoning/` | 两个 Reasoning 源组件及直接清洗、依赖和校准账本 | 否 |
| `data/qa/3.reasoning/hard_expansion/` | 第二批 Reasoning 源组件；旧候选池、pilot 和进度已删除 | 否 |
| `data/qa/7.final_2200/` | 普通、不可回答、Reasoning、Cross-PDF 四个批次无关最终交付 JSON 与分类统计工作簿 | 生成 |
| `data/qa/7.final_2200/rel__collection__final_2200__manifest.json` | 四个正式文件的数量、身份与哈希清单 | 生成 |
| `data/qa/4.cross_pdf/` | 两个 Cross-PDF 源组件及最终选择账本 | 否 |
| `data/qa/4.cross_pdf/hard_expansion/` | 第二批 Cross-PDF 源组件和最终选择账本 | 否 |
| `data/archive/final_2200_cleanup_20260911/` | 按用途分类的历史数据、生成分片、复核记录和评测结果压缩包及 SHA-256 清单 | 是 |
| `data/qa/6.review/` | 直接支持正式发布的 Cross-PDF 人审证据和最终模态复核 | 否 |
| `data/web/` | Web 控制台的账户、审核、任务队列、日志和健康状态 | 是 |
| `data/pdfs/` | 全部 PDF 的唯一平铺目录：`paper_*` 主论文、`source_*` 生成源论文、排序最后的 `z_cross_*` 合订本 | 外部资产 |
| `data/pdf_assets_manifest.json` | 旧路径/旧 ID 到规范 PDF 名称的映射、内容哈希和重复副本审计 | 生成 |
| `data/results/` | 评测运行时创建的推理、Judge、状态和报告输出 | 是 |
| `data/web/review/` | Web 用户/会话/审核 SQLite、内部令牌、审计事件和可逆快照 | 是 |
| `data/web/task_queue/` | 外层任务队列 SQLite、日志、incident 和 Web 健康状态 | 是 |
| `models/` | Qwen、Gemma、Mistral、InternVL、MiniCPM 九套本地权重的唯一真实目录 | 外部资产 |
| `models/.cache/` | Hugging Face、ModelScope、Torch 模型缓存和动态模型模块的唯一缓存根目录 | 外部资产 |
| `sxz/` | 其他同学独立代码；可只读参考，绝对禁止写入或修改 | 不属于本项目 |

最终 2200 条的共享核心字段 contract 位于 `schemas/final_2200_qa.schema.json`；
`src/pku_qa/workflows/selection/normalize_final_2200_release.py` 负责补齐缺失分类、
规范模态别名/顺序并刷新可审计 provenance，`src/pku_qa/workflows/reporting/`
中的 `build_final_2200_classification.py` 生成分类统计工作簿。专项 JSON 可以保留额外
的生成、复审和证据审计字段，但不能缺少共享核心字段或使用非规范取值。
四文件交付使用全局唯一的 `QA0001` 至 `QA2200`：普通题、不可回答题、Reasoning、
Cross-PDF 依次占用连续区间，Cross-PDF 固定排在最后的 `QA1401` 至 `QA2200`。
原 paper/QA ID 保存在 `annotation_provenance.final_2200_identity`。审核数据库已迁移到
四文件和全局 QA ID；事件的 legacy 字段与删除墓碑保留迁移前身份，保证历史可追溯。
最终四文件的 2,200 条模态由
`src/pku_qa/workflows/cleaning/reclassify_cross_pdf_modalities_api.py` 进行可恢复视觉复核；
`build_final_2200_modality_review.py` 准备单 PDF 1,400 条和 Cross-PDF 800 条输入，并在全部
审计完成后统一应用。流程把题目、标准答案和 PDF 页面图像交给 Claude Sonnet 5，严格
校验返回模态和页级依据，再同步四文件正式集与 manifest。
人工终审删除后的两条 Cross-PDF 定额补题已经固化到四文件正式集；一次性补题入口、
候选池和 bundle manifest 已在锁定最终哈希后删除，补题来源身份继续保存在 QA provenance
和最终选择账本中。
`src/pku_qa/workflows/selection/sync_final_2200_manifest.py` 对四文件执行统一 schema、
`1000/200/200/800` 精确数量、旧身份唯一性和 `QA0001..QA2200` 顺序检查，并同步同目录
manifest。Web 注册器、SQLite 审核状态和后续最终评测均使用这四个 dataset ID。

已批准清理的历史内容按用途存入 `data/archive/final_2200_cleanup_20260911/`。每类包含
保留原项目相对路径的 `files.tar.zst` 和记录大小、SHA-256 的 `manifest.csv`；代码、测试、
最终 provenance、任务状态或 SQLite 仍引用的规范路径记录在
`retained_canonical_paths.csv`，并继续原位保留。人工审核快照因撤销事件引用其精确路径，
仍位于 `data/web/review/snapshots/`。

### 5.1 数据文件分组

`data/qa/1.base/` 保存基础谱系，不再把不同阶段平铺到 `data/qa/`：

| 文件 | 用途 |
| --- | --- |
| `work__single_pdf__raw_mixed__n6204.json` | 6204 题原始混合题型数据 |
| `rel__single_pdf__mixed__n4211__v1.json` | 清洗后的 4211 题混合题型正式基线 |
| `rel__single_pdf__short_answer__n4211__v1.json` | 4211 题统一简答正式基线 |
| `rel__single_pdf__short_answer_unanswerable__n4451__v1.json` | 4211 题加 240 道不可回答题的 Hard 集 |
| `work__single_pdf__unanswerable_pool__n240.json` | 240 道不可回答辅助题源 |

其他原始派生、过滤/均衡版本和历史统计表已进入
`data/archive/final_2200_cleanup_20260911/historical_or_upstream_dataset/` 或
`qa_non_json_provenance/`，不再平铺在 `1.base/`。

当前发布与人工审核统一使用 `data/qa/7.final_2200/` 的四个 QA 文件。
Reasoning 的两批 100 与 Cross-PDF 的两批 400 保留为来源组件；它们的旧 ID、题面或证据
可能与人工终审后的全局 QA 不同，不能直接替代最终文件。完整谱系见 `data/qa/README.md`。

`pku_qa.workflows.operations.inspect_final_2200` 是标准库只读入口：复用最终 contract
与 manifest 构建器，检查四文件身份、数量、哈希并重算统计，不要求上游或分类镜像。
`--check-pdfs` 额外使用 pypdf 对最终使用的 712 份 PDF 检查哈希、可读性及证据页上界。
manifest 构建器本身不打开 PDF，因此结构检查与 PDF 资产检查均须通过。

最终集合包含 474 份单论文 PDF 与 238 份合并 PDF；Cross-PDF 621 题依赖两篇、179 题
依赖三篇。普通/不可回答分类镜像只用于单 PDF 专项，由最终文件按字节同步。
消融输入在运行时从普通题逐个移除证据页构造，不是另一个正式发布组件。
历史模型输出、候选与复核材料按用途归档；新的实验仍写入 `data/results/`。

模型的真实文件统一在 `models/<model-id>/`。每个目录中的 config、tokenizer、
processor、generation config 和分片权重属于对应 Hugging Face 模型仓库，不拆散；
`pku_qa.evaluation.model_paths` 从源码位置确定项目根目录，消除当前工作目录对默认路径
的影响；本地 provider 的模型与 processor 路径解析后必须位于 `models/` 内，加载前还会
检查模型目录和 `config.json`。Hugging Face、ModelScope、Torch 的缓存环境变量固定到
`models/.cache/`；项目根目录和用户缓存目录不保留模型副本或兼容链接。

### 5.2 公网 Web 与反向隧道

学校服务器是内网源站，直接访问需要学校 VPN，不能把学校服务器当作公网入口。项目的
公网可用性依赖自有 VPS 的反向映射，因此 Web 相关改动的正确性必须同时在源站和 VPS
公网入口验收：不能只在 localhost 看到页面就结束。任何 Web、代理、登录、路由、静态
资源或服务单元改动，都要保留现有隧道、端口和认证边界，并运行
`PYTHONPATH=src python -m pku_qa.workflows.operations.check_web_stack`；它必须同时确认
本地端点 200、VPS `127.0.0.1:13780` 返回 200，以及公网
`https://pku.chenzijian.com/` 返回 200 并到达应用登录/注册页面。

学校服务器只在 loopback 监听：Web `127.0.0.1:3780`、任务 API
`127.0.0.1:8765`、数据审核 API `127.0.0.1:8770`。现有
`pku-web-tunnel.service` 由学校服务器主动发起出站 SSH 连接，使用专用 key，将 VPS
`45.59.102.64` 的 `127.0.0.1:13780` 反向转发到本地 3780。VPS Nginx 的
`https://pku.chenzijian.com/` 启用 Let's Encrypt TLS，再代理到 13780；用户认证由
应用登录和 HttpOnly session 完成。正式证书 lineage 是
`pku.chenzijian.com-real`，由 Certbot 定时自动续期；规范 Nginx 配置必须引用这个
lineage。公网 401 表示旧 Nginx Basic Auth 错误拦截。公网 DNS 只解析到 VPS，
学校服务器不接受公网入站访问，使用者仍需学校 VPN 才能直接访问学校网内资源。

PDF 通过数据审核 API 的 Range 响应传输，并附带文件变更标识和 24 小时私有缓存头；Next
代理会转发 `Range`、条件请求、`ETag`、`Last-Modified` 和 `Cache-Control`。浏览器缓存
只属于当前校验员，不改变 QA 真源或权限。VPS Nginx 不做跨用户认证响应缓存，但开启响应
缓冲，避免慢公网客户端长期占用反向 SSH 通道；规范配置见
`deploy/nginx/pku.chenzijian.com.conf`。

Nginx 负责 TLS 和反向代理。应用使用 `admin`/`reviewer` 账户、scrypt 密码哈希和
HttpOnly 会话；`czj-web` 也使用应用密码。公网 Next 代理明确拒绝旧 perimeter 交换，
防止客户端伪造外层用户名获得管理员会话；其他同学可以自行注册或由管理员新建账户。
注册和邮箱换绑验证码由数据 API 经 Gmail SMTP 发送；由于学校出口直连 SMTP TLS 不稳定，
服务使用显式 SOCKS5h VPN 连接，而不是依赖 `smtplib` 不会读取的通用 HTTP 代理环境变量。
管理员按 JSON 和 1-based 起止序号创建人工校验范围；范围在 SQLite 中同时保存展示用
边界和稳定的 `dataset_id/paper_id/qa_id` 成员，禁止重叠。五文件迁移留下的分配标记为
`stable_members` 并保持精确成员集合，不能直接改写为连续区间；需要调整时删除后重建。
校验员可读取全部 QA/PDF，
但后端只允许修改自己的分配成员；仅有 admin 身份的账户可全量运维，兼任 reviewer 的管理员仍受本人分配范围限制。所有审核/撤销、分配、账户管理和
任务队列写操作都记录操作者。任务队列入口只向管理员展示；现有任务队列 GET 仍对登录用户可见，写操作仅管理员，人工审核
的原有可逆 JSON 快照逻辑不变。网页修改 `question`、`answer` 或 `evidence_pages` 前会保存
完整文件快照，随后原子写回规范 JSON；证据页会校验为正整数、排序去重并同步
`oracle_pages`、`evidence_span`、`evidence_span_ratio` 及已有 `evidence_items` 页映射。
新增证据页不会虚构 `supported_fact`，删除证据页会删除对应的陈旧 evidence item。
最终四个文件发生修改、删除或撤销后，同一锁内还会刷新
`data/qa/7.final_2200/rel__collection__final_2200__manifest.json` 的组件哈希、题数、论文数和可回答/不可回答计数。

`deploy/systemd/` 保存本地 daemon/API/Web 和两分钟健康检查 timer 的可复现 unit。
`src/pku_qa/workflows/operations/check_web_stack.py` 检查三个本地端点、经 SSH 的 VPS 13780 后端和公网 HTTPS
前置层；`--repair` 只重启失败的本地服务或现有隧道，不修改远端 Nginx、SSH key、
端口或认证。最新状态写入 `data/web/task_queue/web_health.json`。
任务 API 对仪表盘 GPU 状态使用 30 秒缓存，并在缓存刷新时合并并发请求，避免网页的
3 秒刷新重复启动或重叠运行 `nvidia-smi`。

规范输出路径：

- `data/results/evaluations/final_2200/{ordinary,unanswerable,reasoning,cross_pdf}/`：当前四组件评测。
- `data/results/evaluations/single_pdf_1200/`：普通/不可回答专项、动态消融输入和状态。
- `data/results/reports/single_pdf_1200/`：专项严格指标和技术报告。

历史分批输出只用于对应输入的审计，不能覆盖当前四文件结果。

## 6. 金标准数据结构

顶层是 `{paper_id: paper}`，`paper_id` 是与 PDF 文件名对应的字符串。paper 字段：

| 字段 | 含义 |
| --- | --- |
| `paper` | 论文 ID 副本；final_2200 组件必填并应与第一层 key 一致 |
| `primary_category`, `secondary_category` | 学科分类 |
| `pdf_metadata` | 页数、文件或来源元数据 |
| `QA` | `{qa_id: qa}`，一篇论文的题目映射 |

基础 QA 字段：

最终发表前 `final_2200` 的四个 JSON 在下表字段基础上共享一个强制核心 contract：
`question`、`answer`、`evidence_pages`、`modal_types`、`question_type`、
`question_category` 六个字段全部必填。组件可以保留专项生成、复审和证据审计字段，
但不能缺少核心字段或使用非规范取值；完整机器校验定义见
`schemas/final_2200_qa.schema.json`。

| 字段 | 类型与约束 | 含义 |
| --- | --- | --- |
| `question` | 非空字符串 | 模型看到的问题 |
| `answer` | 非空字符串 | 唯一独立金答案；拒答必须精确为 `Unanswerable` |
| `options` | 历史原始/混合文件才可出现 | 旧 MCQ 的 `{id,text}` 选项；最终 2,200 条禁止出现，构建预检会拒绝 |
| `answer_format` | Integer/Float/String/List/Unanswerable（非所有题都有） | 历史构建元数据；公开评测不做类型化预匹配 |
| `answer_aliases` | 数组 | 可接受别名，不改变主金答案 |
| `answer_unit` | 字符串或 null | 数值答案单位 |
| `numeric_tolerance` | null 或 `{type,value}` | absolute/relative 数值容差 |
| `string_metric` | 旧数据可选字符串 | 最终四文件不存在；公开评测不使用字符串预匹配 |
| `evidence_pages` | 唯一正整数数组 | 物理 PDF 页金标；拒答必须为空 |
| `evidence_items` | 数组 | 页内证据对象 |
| `evidence_hops` | 非负整数 | 推理需要的证据跳数 |
| `evidence_span` | 非负整数或 null | 最远证据页跨度 |
| `evidence_span_ratio` | 0..1 或 null | 跨页跨度占论文比例 |
| `modal_types` | 字符串数组；final_2200 必填 | 仅允许按固定顺序使用 `text`、`image`、`table`、`formula`；表示回答实际需要的模态 |
| `question_type`, `question_category` | 字符串；final_2200 必填 | `question_type` 仅为 `Literal` 或 `Inferential`；`question_category` 使用分类说明中的正式问题小类 |
| `annotation_provenance` | 对象 | 构造、来源、审校和版本信息 |
| `review_status` | 非空字符串 | 当前人工/模型审核状态 |

`evidence_items` 核心字段：`physical_pdf_page` 为外部物理页号；
`printed_page_label` 是论文内印刷页号，仅供审计；`source_types` 描述模态；
`quote_or_region` 记录文本摘录或图表区域；`necessity_status` 说明是否为回答必需。

Reasoning 数据额外使用 `evidence_provenance`：记录每个来源 QA 的贡献、来源数据
SHA、PDF SHA/页界、证据合并和双模型审核，以便重建 `evidence_pages`。Challenge
数据额外使用 `oracle_pages`、`input_pages` 和消融变体标识；它们只决定展示页面，
不改变 `pdf` 输出协议。

## 7. 推理结果字段

每个推理 QA 行继承必要的分析元数据，并包含：

| 字段 | 含义 |
| --- | --- |
| `evaluated_model` | 精确被测模型标识 |
| `question`, `correct_answer`, `type` | 与独立金标绑定的问题、答案、题型 |
| `model_output` | 允许的确定性规范化后输出；闭卷保持原样 |
| `raw_model_output` | 模型逐字输出 |
| `raw_model_output_sha256` | 原始输出哈希 |
| `deterministic_normalizations` | 发生的允许规范化列表 |
| `input_mode` | `question_only` 或 `pdf` |
| `reference_evidence_pages` | 独立金证据页副本 |
| `require_structured_output`, `require_evidence_pages` | 由协议和是否拒答派生 |
| `shown_pdf_pages` | 实际展示给模型的物理页；闭卷为空 |
| `total_pdf_pages`, `max_pdf_pages` | PDF 实际总页与用户页上限 |
| `requested_pdf_dpi`, `rendered_pdf_dpi`, `pdf_render_downscaled` | 像素预算审计 |
| `page_input_policy`, `prompt_style` | 页面选择和提示类型 |
| `protocol_fingerprint` | 代码、provider、参数和协议 contract 哈希 |
| `qa_source_sha256` | 整个 QA JSON 哈希 |
| `pdf_corpus_sha256`, `pdf_sha256` | PDF 清单聚合哈希与本篇 PDF 哈希 |
| `generation_status` | 可选异常生成状态；不把非法输出变成答案 |

## 8. 内部历史 Judge 结果字段

Judge 行复制所有用于绑定的推理字段，另加：

| 字段 | 含义 |
| --- | --- |
| `parsed_answer` | 从合法 PDF JSON 解析的答案，或闭卷原文 |
| `predicted_evidence_pages` | 模型预测页集合；闭卷为空 |
| `reference_evidence_pages` | 金页集合 |
| `output_parse_method` | `canonical_json`、`question_only_plain_answer` 或非法原因类别 |
| `output_is_legal`, `output_illegal_reason` | 输出 contract 合法性 |
| `answer_is_correct` | 只看答案的布尔分数 |
| `evidence_pages_is_correct` | 页集合完全一致；闭卷为 true |
| `is_correct` | 合法且答案、证据均正确的联合分数 |
| `match_method` | exact/alias/numeric/choice/LLM Judge 等实际路径 |
| `typed_score` | 类型化评分诊断，可为空 |
| `judge_verdict`, `judge_response` | 规则或 27B Judge 结论与可选原文 |
| `inference_binding_sha256` | 该 Judge 行绑定的精确推理行哈希 |
| `judge_protocol_fingerprint` | Judge 代码、模型身份和参数哈希 |
| `publication_eligible` | 正式代码恒为 true；缺失即拒绝报告 |
| `protocol_validation` | 正式代码恒为 `publication_strict` |

内部历史报告的三项诊断分别聚合 `answer_is_correct`、`evidence_pages_is_correct` 和
`is_correct`。非法输出保留在固定全集分母中，另报告非法率，不能从分母剔除。

## 9. 断点和任务字段

内层 `.adaptive_queue/<phase>/` 包含 source/manifest、逐论文结果、claim/failure、
scheduler state 和合并产物。contract 或 required field/value 不匹配时，旧结果不能
导入；匹配时可从完成的论文继续。claim 先完整写入同目录临时文件，再以原子硬链接
抢占正式路径，其他工作进程不会观察到半写 JSON 或重复领取同一论文。

外层 `Task` 字段分组：

- 身份/命令：`id`, `name`, `command`, `cwd`, `env`, `kind`, `metadata`。
- 排序/依赖：`status`, `position`, `priority`, `depends_on`, `enabled`, `adopted`。
- 进程：`pid`, `runner_pid`, `tmux_session`, `log_path`。
- 重试：`max_restarts`, `restart_count`, `retry_delay_seconds`, `next_retry_at`,
  `timeout_seconds`。
- 活性/进度：`heartbeat_at`, `progress_heartbeat_at`, `progress_signature`,
  `progress_path`。
- 完成验证：`validation_command`。
- 恢复控制：自动恢复开关、尝试上限、当前尝试次数和错误记录。
- 时间：`created_at`, `updated_at`, `started_at`, `finished_at`。

状态 `starting/running/waiting/recovering` 为活动态；
`succeeded/cancelled/blocked` 为终态。资源类为 `cpu/gpu/gpu_high_vram`。
删除表和 incident 表只记录当前队列的可审计操作；无效任务应受控删除，已成功且可
断点恢复的任务保留原 ID，并在目录迁移时事务更新命令和路径。

## 10. 测试和文档维护

`PYTHONPATH=src python -m pytest -q` 覆盖纯函数单元测试、数据 contract、Reasoning 证据恢复、
Challenge 预检、端到端 Judge/Report、防篡改指纹、断点队列、GPU 调度、人工审核和
文档清单。`tools/internal/experiment_console/web` 中的 `npm test` 先构建再检查渲染 HTML。

`docs/current/PROJECT_INVENTORY.md` 是不含 `sxz/` 的逐文件哈希清单。任何代码变化后，
都应重新生成清单并同步本文件及上手指南。
