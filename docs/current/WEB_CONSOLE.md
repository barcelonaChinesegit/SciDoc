# 本地任务与人工校验控制台

`tools/internal/experiment_console/web/` 是最终 2,200 条 QA 的多人校验入口，同时保留任务队列、GPU、
断点和日志控制。源码位于内部维护工具目录，不是 benchmark 使用依赖。任务队列导航和首页卡片仅向管理员展示；根路径是审核入口封面：

学校服务器是内网源站，直接访问需要学校 VPN；公网用户只能通过项目自有 VPS 的反向
隧道访问。公网入口 `https://pku.chenzijian.com/` 是 Web 控制台的正式验收地址，DNS
当前指向 VPS `45.59.102.64`。请求链路固定为
`pku.chenzijian.com:443` → VPS Nginx → `127.0.0.1:13780` → 反向 SSH 隧道 →
学校 Web `127.0.0.1:3780`；学校服务器无需也不得开放公网入站端口。
网页、代理、登录、路由、静态资源或服务配置发生任何改动后，必须确认本地
`127.0.0.1:3780` 与 VPS `127.0.0.1:13780` 均可用，再从公网入口检查同一页面或资源；
只在内网浏览器打开成功不能作为交付依据。

- `/data`：逐题人工校验、问题/答案/证据页修改、QA 序号/PDF 物理页跳转、可逆删除和撤销。
- `/guide`：英文人工校验指南，覆盖权限边界、四类正式数据、逐题流程和故障处理。
- `/queue`：维护者实验任务、GPU、进度、日志和断点；仅管理员看到导航入口。
- `/admin/users`：管理员创建/停用用户、分别添加/删除管理员和校验员身份、重置密码、分配 QA 范围并查看统一审计。
- `/profile`：个人中心；用户可修改真实姓名、用户名、验证邮箱和密码，邮箱换绑需要验证码。
- `/login`、`/register`：应用账户登录和公开注册；注册账户默认只有校验员身份。

当前 Web 界面采用英文单语，包括登录、导航、人工校验、用户管理、个人中心、任务队列、
错误提示和无障碍文本。`tools/internal/experiment_console/web/tests/rendered-html.test.mjs` 会递归扫描 `app/` 下的
源码并拒绝汉字，防止后续界面文案回退为中文。数据 API 返回的目录元数据在前端转换为
英文展示文本；QA 问题、标准答案和证据事实仍按数据真源原样呈现。账户姓名、任务名称、
任务说明和日志若含汉字，则由展示层改用英文用户名或英文占位，不改写数据库与审计真源。

访问公网根地址时，未登录用户会直接看到整页登录入口，而不是弹窗或居中的小卡片。登录
页采用响应式分栏布局：左侧显示本批次和权限边界，右侧填写管理员分配的用户名与密码；
手机屏幕会自动改为上下排列。浏览器已有有效 `pku_session` 时，根地址会直接进入封面，
从受保护页面跳转来的用户登录后会回到原页面。登录页先尝试已有应用会话，无有效会话时
直接显示账户密码或注册表单。VPS 不再使用会触发浏览器白色弹窗的 Nginx Basic Auth，
`czj-web` 与其他账户一样使用应用密码和同一枚 HttpOnly session cookie。

### PDF 缓存与中转层

PDF 响应使用 `ETag`、`Last-Modified` 和 `Cache-Control: private, max-age=86400`。
因此每位校验员的浏览器会在本机保留已查看的 PDF，重新打开同一论文或跳转物理页时优先
复用本地缓存；文件发生变化或缓存过期后，浏览器才发条件请求。JSON、会话和审核 API
仍然禁止缓存。点击证据物理页时，页面会用目标 `#page=N` 重新挂载 PDF iframe；这是因为
部分 Chrome/Edge 内置 PDF 查看器不会响应现有 iframe 的纯 fragment 变化。重复点击当前页
也会重新定位，但底层 PDF URL 不变，浏览器继续复用私有缓存，不需要重新传输整份文件。

VPS Nginx 保持跨用户 `proxy_cache off`，但开启 `proxy_buffering`，让慢客户端不会持续
占用学校服务器到 VPS 的 SSH 通道；HTTPS 同时启用 HTTP/2、文本响应 gzip、到 loopback
隧道后端的 HTTP/1.1 连接复用和 TCP keepalive。可复现配置位于
`deploy/nginx/pku.chenzijian.com.conf`；安装或修改后必须先运行 `nginx -t`，再 reload，
并执行项目公网健康检查。不要重新添加会遮挡应用登录页的 Basic Auth，也不要把学校源站
或 loopback API 直接暴露到公网。生产 TLS 使用 Let's Encrypt lineage
`pku.chenzijian.com-real` 并由 VPS Certbot 自动续期；不要改回 bootstrap 自签名证书，
也不要直接用 VPS IP 代替域名访问。

### 目录与交互性能

`data/qa/` 保存最终文件、保留的上游来源及复核材料；大规模历史 checkpoint 和模型响应已归档。人工校验目录识别其中
实际含 `QA` 映射的 JSON。数据 API 因此在启动时只预热四个正式 2200 文件的注册器，并将目录结果保留
5 分钟；网页的编辑、删除和撤销会立即使对应缓存失效，所以通过网页改动的 JSON 会在下一次
目录读取中立刻反映，外部进程直接写入的文件最迟在 5 分钟后重新发现。不要为此把过程目录
改成不含 QA 的假结构或改动正式数据命名。

`/data-api/datasets?collection=final_2200&detail=summary` 只返回侧栏和分配下拉框需要的正式
2200 集合文件名、数量、集合和权限，
不再重复传输所有 JSON 的字段覆盖率；选中某个数据集后，网页再请求
`/data-api/datasets/<dataset-id>` 取得该文件完整的来源、用途、统计和字段字典。因此管理员
改变身份、创建分配、翻到下一题、保留、修改、删除或撤销时，都不会重新下载整个目录。当前
QA 和相邻上一/下一题会在浏览器中保留一个小型短期缓存；提交写操作后该缓存立即清空并重新
读取服务器结果，不能把此缓存当作可编辑数据真源。

“其他数据与历史版本”在网页初始状态不会展示，也不会触发历史目录扫描。只有点击侧栏底部
的展开箭头后，网页才请求 `/data-api/datasets?collection=other&detail=summary`，后端才扫描
并返回其余可识别为 QA 的 JSON；再次点击箭头可以收起已经加载的结果。

人员账户中的绿色“校验员”表示已拥有 reviewer 身份，蓝色“管理员”表示已拥有 admin 身份，
灰白表示该身份尚未添加。点击后先在当前页面立即反映，按钮短暂显示“保存中…”并禁止重复
提交；服务器确认后写入用户审计记录。`czj-web` 的管理员身份是 bootstrap 保护项，不能删除。

登录后的校验员会在 `/data` 页顶部看到“我需要校验的题目”面板。新建的连续区间显示
1-based 起止序号；从五文件迁移来的任务显示为固定成员集。两者都显示总题数、已完成数以及
保留/修改/删除的细分进度；点击卡片会直达第一个分配成员。个人中心 `/profile` 也会显示同一份分配摘要。前端通过
`/data-api/assignments?scope=mine` 获取数据，后端只返回当前 session 对应账户的分配，不会
暴露其他同学的范围。管理员继续通过不带 `scope=mine` 的接口查看和编辑全部分配。

## 认证与权限

VPS Nginx 只负责 HTTPS 和到 loopback 隧道后端的反向代理。应用自身在
`data/web/review/review.sqlite3` 保存用户、scrypt 密码哈希、14 天 HttpOnly
会话和审计事件，不保存明文密码。

管理员在 `/admin/users` 的“校验员进度监督”区域可以看到每个拥有 reviewer 身份的账户，
包括没有分配任务的账户。统计按稳定 QA 成员计算，显示已完成/已分配、完成率、保留、修改、
删除和最近一次完成时间；停用账户会保留历史进度并明确标记。该区域只供管理员查看，不改变
原有分配权限或审核审计记录。

`czj-web` 是不可停用、不可降级的 bootstrap 管理员，并已配置独立应用密码。公网不再
开放 perimeter 身份交换；Next 代理会拒绝 `/data-api/auth/perimeter`，避免客户端伪造
外层用户名换取管理员会话。所有用户都通过应用登录/注册页面建立 session。

账户身份以 `roles` 集合保存，`admin` 和 `reviewer` 可以同时存在。同一账户可同时拥有管理员
管理权限和校验员工作权限；网页用两个独立身份按钮添加或删除身份，至少保留一个身份，
`czj-web` 的管理员身份不可删除。带 `reviewer` 身份的账户可浏览全部数据、PDF 和审核日志，
但只能保留、修改、删除或撤销管理员分配给自己的 QA；这条限制同样适用于同时拥有
`admin + reviewer` 的兼任账户。未分配题在网页中明确显示为只读，写请求也会在数据 API
返回 403。只有 `admin`、未兼任 reviewer 的管理账户保留原有的全量 QA 运维权限；需要参加
正式分摊时应先添加 reviewer 身份并建立明确分配。拥有 `admin` 身份的账户可按 JSON 与
1-based 起止序号新建、调整或删除分配。迁移后的 `stable_members` 分配保持原 2,200 条精确
成员，不能直接改成一个新的连续范围；管理员需要先删除再重建。创建分配时
系统会把范围固化为 `dataset_id + paper_id + qa_id` 成员清单，因此前序题被删除后不会
把后续题的写权限错误地移给其他人。不同分配不能覆盖同一 QA。

任务队列入口仅管理员可见；已有读 API 权限仍向登录用户开放，新增、编辑、暂停、恢复、重试、排序、删除、
结果删除和恢复记录仅管理员可用。

## 审计与真源

人工校验事件保存操作者 user ID/username、数据集、paper/QA ID、动作、前后 JSON
SHA-256、修改或删除前的完整快照、时间和撤销人。账户登录/失败登录/退出、新建、角色或状态
修改、密码重置，以及 QA 分配的新建、调整和删除也写入不可变审计表。

任务队列 SQLite 的 `task_audit_events` 记录网页代理校验过的操作者及所有队列写动作。
daemon/CLI 的内部动作仍可作为 `system` 运行，不伪装成人工账户。

QA JSON 仍是实验真源。`edit` 和 `delete` 都会先在
`data/web/review/snapshots/` 保存完整文件，再原子写回正式 JSON；`undo` 只允许
撤销当前文件哈希仍匹配的最近操作。编辑证据页时，后端校验正整数与 PDF 页界、排序
去重，并同步 `oracle_pages`、跨度字段和已有 `evidence_items` 的物理页映射；新增页不
虚构证据事实。最终文件修改、删除或撤销后还会同步刷新
`data/qa/7.final_2200/rel__collection__final_2200__manifest.json` 的哈希与计数。用户、会话和审计数据库不是第二份 QA。

final_2200 四个文件的 QA 核心字段已统一：`question`、`answer`、`evidence_pages`、
`modal_types`、`question_type`、`question_category` 必须全部存在；模态只允许
`text`、`image`、`table`、`formula`，题型只允许 `Literal`/`Inferential`。字段字典中看到
的其它字段是组件专用的来源、生成、复审或证据审计元数据，网页应保留它们以支持追溯。
统一 contract 的 schema 是 `schemas/final_2200_qa.schema.json`，集合检查命令是：

```bash
PYTHONPATH=src python -m pku_qa.workflows.selection.normalize_final_2200_release --check-only
PYTHONPATH=src python -m pku_qa.workflows.selection.sync_final_2200_manifest --check-only
```

公开注册要求真实姓名、唯一用户名、至少 10 位密码和邮箱验证码。验证码只保存带盐哈希，
10 分钟有效、60 秒内不可重复发送；SMTP 凭据只在服务端被忽略的环境文件中提供，绝不进入
网页或审计日志。个人中心换绑邮箱同样需要对新邮箱发送验证码；修改密码会注销其他设备会话。
学校出口与 Gmail SMTP 的 TLS 握手可能超时，因此部署服务通过
`PKU_SMTP_PROXY=socks5h://127.0.0.1:10808` 明确使用本机 VPN。`smtplib` 不会自动采用
`HTTP_PROXY`/`ALL_PROXY`，不能只在 shell 中设置代理。代理不可用时接口返回发送失败，并在
服务日志中只记录异常类型，不记录邮箱、验证码或 SMTP 密码。

完整操作标准见 [人工校验手册](MANUAL_REVIEW_GUIDE.md)。

## 本地运行与测试

```bash
cd tools/internal/experiment_console/web
npm install
npm run dev
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY npm test
```

Web 将浏览器请求代理到任务 API `127.0.0.1:8765` 和数据审核 API
`127.0.0.1:8770`。两个后端都必须保持 loopback-only。

部署验收使用项目健康检查：

```bash
PYTHONPATH=src python -m pku_qa.workflows.operations.check_web_stack
```

该命令会检查本地三个端点、VPS 反向隧道后端和公网 HTTPS 前置层。公网必须返回 200；
401 表示旧 Basic Auth 仍在拦截应用登录页，应视为失败。

数据集目录中的旧 Reasoning 100、4,211 题基线和 983 题人工审核来源明确标为历史构建材料。当前评测与终审只使用 final_2200 四文件；历史数据的既有权限由注册器独立保留。

登录页与封面统一显示四个正式 QA 文件、2,200 道题。界面参考见 [人工审核手册](MANUAL_REVIEW_GUIDE.md#18-当前界面参考)，图片随文档提供。
