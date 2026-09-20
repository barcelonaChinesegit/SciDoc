# Web 控制台运行数据

本目录集中保存网站及其后端服务产生的持久记录，不存放前端源码，也不属于正式 QA 或 PDF 数据。

| 目录 | 内容 |
| --- | --- |
| `review/` | 用户、会话、角色、分配、人工审核事件、内部代理令牌和可撤销 JSON 快照。 |
| `task_queue/` | 任务队列 SQLite、daemon 恢复状态、任务日志、incident、锁和 Web 健康检查结果。 |

`review/review.sqlite3` 中的审核事件通过绝对路径引用 `review/snapshots/`。移动本目录时必须同步迁移这些路径，才能继续撤销历史操作。

`task_queue/tasks.sqlite3` 和 `task_queue/daemon_state.json` 保存任务日志、进度及恢复路径。服务默认路径和 systemd 单元必须始终与本目录一致。
