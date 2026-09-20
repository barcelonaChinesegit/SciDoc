# SciDoc 上手指南

本机唯一实际工作目录为 `/data/czj/SciDoc`，远程仓库为
<https://github.com/barcelonaChinesegit/SciDoc>。首次克隆使用：

```bash
git clone https://github.com/barcelonaChinesegit/SciDoc.git
cd SciDoc
```

2026-09-20 的完整本机迁移记录见 [迁移验收](../reports/SCIDOC_MIGRATION_20260920.md)。
本地资产、密钥、数据库及只读协作目录 `sxz/` 都在新目录完整保留，但不提交公开 Git。
旧 `/data/czj/pku` 仅是指向新目录的符号链接，供不可改写的历史快照和协作脚本解析路径；
请在新目录开发，新服务和新任务均使用新路径。

## 1. 首次检查：无需 GPU

使用 Python 3.12+，在仓库根目录执行：

```bash
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200
```

只读检查输出 `status: valid`、2,200 QA、四组件 `1000/200/200/800`，以及
712 份评测 PDF 的引用数。此时只验证 JSON、共享字段、身份、数量和 manifest 内容哈希，
不需要 PDF、分类镜像、上游基线、API 密钥或第三方 Python 包。

最终文件位于 `data/qa/7.final_2200/`，分别是 `ordinary_qa.json`、
`unanswerable_qa.json`、`reasoning_qa.json`、`cross_pdf_qa.json`。
上游 4,211、Reasoning 两批 100、Cross-PDF 两批 400 是构建来源，不是当前评测入口。

## 2. 准备环境与资产

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

也可使用 `conda env create -f environment.yml` 后 `conda activate pku-qa`。
仅检查 PDF 可只安装 `pypdf==6.10.2`；完整 requirements 用于模型推理、API 和测试。

Git 中提供最终 QA 和资产索引，PDF、权重、历史归档压缩包、审核数据库与密钥单独管理。
向维护者取得与 `data/pdf_assets_manifest.json` 匹配的 PDF，平铺到 `data/pdfs/`。
最终输入使用其中 474 份单论文和 238 份合并 PDF；完整资产目录还包含生成源论文和历史合订本，
不能把资产总数写成 benchmark 的论文数。

```bash
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200 --check-pdfs
```

成功时 `pdf_check` 为 `hashes_readability_and_evidence_bounds_valid`。这一步逐一验证
最终 PDF 的 SHA-256、可读性及全部金证据页上界。只检查 manifest 不会打开 PDF。
需要核对整个资产目录而非最终使用子集时，运行：

```bash
PYTHONPATH=src python -m pku_qa.workflows.operations.flatten_pdf_assets --check
```

默认受控评测需要 CUDA 和以下模型：

- `models/Qwen3-VL-4B-Instruct/`
- `models/Qwen3-VL-8B-Instruct/`
- `models/Qwen3.6-27B/`（统一语义 Judge）

其他本地模型仅在对应实验使用时准备。完整权重和 processor 必须位于 `models/` 内，
并包含模型自己的 config/tokenizer 等文件。Hugging Face、ModelScope、Torch 缓存统一在
`models/.cache/`。根据实际 GPU 调整命令，动态 A800 配置不是任意 CUDA GPU 的通用配置。

## 3. 四文件正式评测

先做第 1、2 节的只读预检，再打印 manifest 驱动计划：

```bash
PYTHONPATH=src python -m pku_qa.workflows.reporting.run_final_2200_evaluation \
  --print-plan -- --dynamic-a800 --a800-gpus 2 3 4 5
```

打印计划不启动 GPU 工作或外部 API 调用。运行器锁定 `--qa-json`、`--pdf-dir`、
`--output-dir`、`--input-mode pdf`、数量及 QA SHA-256。移除 `--print-plan` 即执行：

```bash
PYTHONPATH=src python -m pku_qa.workflows.reporting.run_final_2200_evaluation \
  -- --dynamic-a800 --a800-gpus 2 3 4 5
```

单组件例子：

```bash
PYTHONPATH=src python -m pku_qa.workflows.reporting.run_final_2200_evaluation \
  --component reasoning --print-plan -- --dynamic-a800 --a800-gpus 2 3 4 5
```

`--component` 可重复；缺省按 ordinary、unanswerable、reasoning、cross_pdf 顺序执行。
每个组件执行 4B → 8B → Judge 4B → Judge 8B → 严格报告，输出在
`data/results/evaluations/final_2200/<component>/`。推理负责记录 PDF corpus 哈希与协议指纹，
正式报告重新核验 QA/PDF、原始输出和推理到 Judge 的绑定。

## 4. 协议、对照和专项实验

只有两种协议：

| 协议 | 输入 | 输出 |
| --- | --- | --- |
| `question_only` | 只有问题，不读取 PDF | 答案原文 |
| `pdf` | 问题与标注 `[Page N]` 的 PDF 页图 | `{"answer_pre":"...","evidence_pages":[1,3]}` |

Full、Oracle、`qa_field` 消融仅决定 `pdf` 页选择。不可回答标签必须精确为
`Unanswerable`，PDF 拒答证据页为 `[]`；不要加“是否要求证据页”的独立开关。

需要闭卷对照时，直接使用通用运行器，并为不同协议、页面策略使用独立输出目录：

```bash
PYTHONPATH=src python -m pku_qa.evaluation.run_hard_eval \
  --qa-json data/qa/7.final_2200/reasoning_qa.json \
  --output-dir data/results/evaluations/reasoning_200_question_only \
  --input-mode question_only --page-input-policy full \
  --expected-qa-count 200 \
  --dynamic-a800 --a800-gpus 2 3 4 5
```

正式运行还应传入已冻结的 `--expected-qa-sha256`；PDF 模式另冻结
`--expected-pdf-corpus-sha256`。具体 provider、显存门槛和重试参数以
`PYTHONPATH=src python -m pku_qa.evaluation.run_hard_eval --help` 为准。

单 PDF 专项包含普通题闭卷、1,200 题 Full/Oracle 和普通题证据页消融。先从最终文件同步
两个分类镜像，再使用专项编排器：

```bash
PYTHONPATH=src python -m pku_qa.workflows.selection.sync_final_2200_manifest
PYTHONPATH=src python -m pku_qa.workflows.reporting.run_single_pdf_challenge_gpu_queue \
  --expected-ordinary-sha256 '<ordinary_qa.json 的 SHA-256>' \
  --expected-unanswerable-sha256 '<unanswerable_qa.json 的 SHA-256>'
```

此同步命令会更新 manifest 及只读镜像；首次浏览不需要运行它。专项从最终普通题实时构造
逐页移除的消融输入，消融数量随金证据变化，不应固定假设为历史 850 题。
专项状态在 `data/results/evaluations/single_pdf_1200/state.json`，报告在
`data/results/reports/single_pdf_1200/`。历史 `Challenge119` 是旧 1,190 题的抽样，不能替代
当前 1,200 或 2,200 题报告。

## 5. 数据维护与统计

维护者更新金标准后执行：

```bash
PYTHONPATH=src python -m pku_qa.workflows.selection.normalize_final_2200_release --check-only
PYTHONPATH=src python -m pku_qa.workflows.selection.sync_final_2200_manifest
PYTHONPATH=src python -m pku_qa.workflows.selection.sync_final_2200_manifest --check-only
PYTHONPATH=src python -m pku_qa.workflows.reporting.build_final_2200_classification
PYTHONPATH=src python -m pku_qa.workflows.reporting.build_final_2200_classification --check-only
```

规范化检查可能依赖外部上游 4,211 题源；它不属于干净克隆的首次检查。
工作簿位于最终目录的 `final_2200_classification_statistics.xlsx`，包含四组件、领域、
题型、模态、跨页、文档数、视觉审核和逐题明细。统计只从最终四文件计算。

Reasoning/Cross-PDF 的生成、双审、证据恢复与难度筛选入口保留在
`src/pku_qa/workflows/{generation,review,cleaning,selection}/`。先阅读对应模块 `--help`
与数据目录 README；已清理的大候选池、pilot、旧 provider 输出不能作为即用命令的前提。
历史产物按 `data/archive/final_2200_cleanup_20260911/` 分类保存，恢复时先解包到独立
工作目录核对原路径和 SHA-256，不在当前仓库根目录覆盖解包。

视觉模态复核由 `build_final_2200_modality_review --prepare` 准备输入，
`reclassify_cross_pdf_modalities_api` 执行 API 审核，最后由
`build_final_2200_modality_review --apply` 在 2,200 条全部成功后统一应用。
当前四文件已完成该审核；这是维护流程，会产生 API 成本和数据写入，不是 Quick Start。

## 6. 人工审核和 Web

打开 [公网控制台](https://pku.chenzijian.com/)，注册或使用个人账户登录。
当前界面为英文；本项目中文手册解释其按钮语义。
管理员在 `/admin/users` 分配 QA，校验员在 `/data` 的 **My Review Assignments**
及 `/profile` 查看自己的稳定成员范围；未分配题只可查看。
四步流程为：读问题答案 → 逐页核验证据 → 检查题型依赖 → 保留、修改、删除或暂缓。
完整标准见 [人工审核手册](MANUAL_REVIEW_GUIDE.md)。

本地开发需要 Node 22.13+：

```bash
cd task_queue_web
npm ci
npm run dev
```

任务 API 为 `127.0.0.1:8765`，数据 API 为 `127.0.0.1:8770`，Web 为
`127.0.0.1:3780`。包路径为 `pku_qa.services.task_queue` 与 `pku_qa.services.review`；
账户、会话及可逆快照在 `data/web/review/`，任务状态在 `data/web/task_queue/`。
部署单元见 `deploy/systemd/`，账户和 SMTP 配置见 [Web 控制台](WEB_CONSOLE.md)。

学校源站是内网服务，直接访问需要学校 VPN。公网链路固定为
`https://pku.chenzijian.com/` → VPS `45.59.102.64` Nginx → VPS `127.0.0.1:13780`
→ SSH 反向隧道 → 学校 `127.0.0.1:3780`。保留现有 key、映射、loopback、HTTPS、
应用登录和 HttpOnly session；证书 lineage 为 `pku.chenzijian.com-real`，由 Certbot 续期。
公网不能增加 Nginx Basic Auth，否则会挡住应用登录注册页。

## 7. 长任务和故障排查

```bash
PYTHONPATH=src python -m pku_qa.services.task_queue.task_queue_cli --help
PYTHONPATH=src python -m pku_qa.services.task_queue.task_queue_cli list
PYTHONPATH=src python -m pku_qa.services.task_queue.task_queue_cli show <TASK_ID>
PYTHONPATH=src python -m pku_qa.services.task_queue.task_queue_cli logs <TASK_ID>
```

部署使用 `pku-task-queue-daemon.service` 时由 systemd 管理，不要再启动第二个 CLI daemon。
长任务设置仓库工作目录、资源类、依赖、验证命令及 scheduler/state 进度路径。
只有代码、数据、模型和参数 contract 一致，断点才能恢复。

| 现象 | 检查 |
| --- | --- |
| 找不到 PDF | 最终 712 份资产是否按清单部署；Git 不含 PDF |
| manifest 或 ID 检查失败 | 核对当前四文件与人工修改；不要用旧批次覆盖 |
| GPU 等待 | `nvidia-smi`、A800 类型、GPU 编号与显存门槛 |
| 推理停滞 | `scheduler_state.json` 心跳、`failures/`、OOM 和 provider 日志 |
| 无法复用断点 | 检查数据、协议、参数和输入绑定是否变化 |
| 公网 401 | VPS 是否仍启用旧 Nginx Basic Auth |
| 403 或页面只读 | 登录身份和当前题的稳定分配成员 |

非法输出经过有限次格式纠错后保留原文及哈希，由 Judge 计错，不无限重试。
严格报告需要完整题目键集合；缺失结果应补齐运行，不得只按成功样本重算分母。

迁移后路径检查结果见 [路径与引用审计](../reports/SCIDOC_PATH_AUDIT_20260920.md)。
Web 构建会重新定位 `.vinext/fonts` 缓存中的磁盘路径，避免搬迁后字体 URL 返回 404；
`task_queue_web/build/` 中的手写构建辅助文件必须随源码提交。
迁移回归测试会在独立临时目录验证正式数据检查与评测计划，不依赖本机 editable 安装。

## 8. 修改后的验证

```bash
python -m pytest -q
```

Web 或其后端变更还要执行 Web 构建测试、重启相关本地服务，再验收公网：

```bash
(cd task_queue_web && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy npm test)
systemctl --user restart pku-task-queue-web.service
PYTHONPATH=src python -m pku_qa.workflows.operations.check_web_stack
```

健康检查必须确认三个本地端点、VPS 隧道后端及公网应用页面均返回 200；localhost 成功
不足以完成 Web 变更。`sxz/` 始终严格只读。
