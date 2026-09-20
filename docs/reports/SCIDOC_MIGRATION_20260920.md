# SciDoc 本机目录与 GitHub 仓库迁移（2026-09-20）

> 历史记录：文中旧路径与当时测试结果保留用于追溯。当前 Web 源码在
> `tools/internal/experiment_console/web/`；当前论文协议和复现结论见
> [2026-09-20 评测审计](official_evaluation/FINAL_REPORT.md)。

实际工作目录：`/data/czj/SciDoc`。远程仓库：
<https://github.com/barcelonaChinesegit/SciDoc>。

## 文件保全

迁移通过同一文件系统的目录重命名完成，没有按 Git 跟踪清单筛选本机文件。
迁移后逐项比较设备号、inode、类型、权限、所有者、大小、mtime 与符号链接目标，
155,320 个条目全部一致，缺失或差异为 0：148,923 个普通文件、6,323 个目录、
74 个符号链接；普通文件逻辑大小共 363,986,205,095 字节。
这是迁移切换时的快照，随后服务日志、缓存和构建产物会继续正常变化。

隐藏环境文件、密钥目录、模型及其缓存、全部 PDF、历史归档、论文材料、
Web 账户/会话/审核数据库、快照及 `sxz/` 全部保留。
`sxz/` 通过上层目录整体改名迁入，其内部文件没有编辑、删除或重新生成。

原仓库 `.git` 完整保存在本机 `.migration-backup/pku.git/`，包括未提交前的
index、分支、对象与 reflog；当前工作文件保留迁移前所有已保存的修改。
新根目录 `.git` 使用用户克隆的 GitHub 空仓库，首次提交不携带旧仓库历史中的
大文件或敏感运行材料。备份和逐文件迁移清单不提交公开仓库。

本地审计文件：

- `.migration-backup/before.json`：全部条目的迁移前元数据，含 2,780 个文件的 SHA-256。
- `.migration-backup/move-verification.json`：整体移动后的逐项比较结果。
- `.migration-backup/database-before.json`：三个 SQLite 数据库的完整性和各表行数。
- `.migration-backup/systemd/`：迁移前已安装服务单元备份。
- `.migration-backup/` 中的测试日志和后续验收结果：仅本机保留。

## 新目录运行

源码内的服务工作目录、Python 包路径、数据库路径、环境文件路径与 Web 新任务
默认工作目录已改用 `/data/czj/SciDoc`，本机安装的 systemd 单元同步更新。
四个 ModelScope 绝对缓存链接改为相对链接，模型可独立从新目录访问。
原有 VPS 隧道、密钥、端口、HTTPS 与应用登录边界保持不变。

`/data/czj/pku` 仅保留为指向 `/data/czj/SciDoc` 的目录符号链接，没有第二份数据。
它允许协作脚本及不可改写的历史记录继续解析旧绝对路径；新开发、部署和任务
统一使用新路径。不可在没有检查历史引用及 `sxz/` 使用方的情况下删除该链接。

## 验收结果

- 整体移动的 155,320 个条目：缺失和元数据差异均为 0。
- 迁移后的 2,778 个 SHA-256 复核：无非预期内容变化；修改仅来自迁移配置、文档和正常运行状态。
  另两项为 SQLite 关闭时已完成 checkpoint 并清除的临时 WAL/SHM 文件，不是独立业务数据。
- `sxz/` 的 106,845 个条目再次核对设备号、inode、类型、权限、所有者、大小和 mtime：差异为 0。
- 三个 SQLite 数据库完整性均为 `ok`，全部表行数与迁移前一致。
- 74 个原有符号链接均可解析；Python editable 安装已更新到新 `src/` 路径。
- Python：423 项测试通过。Web：构建成功，6 项测试通过。
- 正式发布集：2,200 QA、四组件 `1000/200/200/800` 校验通过；712 份 PDF 的哈希、可读性和证据页范围通过。
- 重启后的最终健康检查：三个本地端点、VPS 隧道后端和公网入口全部返回 200，`healthy: true`。
- 拟发布文件凭据模式扫描无命中，最大文件低于 GitHub 普通文件的 100 MiB 上限。

## Git 公开范围

新仓库提交源码、测试、部署模板、项目文档、资产索引和正式四文件 QA 发布集。
`.gitignore` 将模型、PDF、非发布数据、运行结果、账户/会话数据库、密钥、环境文件、
构建与依赖缓存、旧 Git 备份、论文工作材料及协作者 `sxz/` 留在本机。
忽略只控制 Git 提交，不意味着本机迁移遗漏或删除这些内容。

## 验收命令

```bash
cd /data/czj/SciDoc
python -m pytest -q
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200 --check-pdfs
(cd task_queue_web && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy npm test)
PYTHONPATH=src python -m pku_qa.workflows.operations.check_web_stack
```

推送须使用有仓库写权限的 GitHub SSH key 或 HTTPS Personal Access Token；
GitHub 不支持账户密码进行 Git 推送。迁移和本地提交不等同于推送成功，必须以
`git ls-remote origin refs/heads/main` 与本机提交一致为远端验收依据。
