# SciDoc 迁移后路径与引用审计（2026-09-20）

## 修复结果

1. **Web 字体实际返回 404。** vinext 的 `.vinext/fonts/*/style.css` 保存了迁移前的
   绝对磁盘路径；构建只替换当前目录前缀，因此旧路径进入 HTML 的字体 URL。
   新 `task_queue_web/build/font-cache.mjs` 在构建配置加载时，根据实际缓存目录重新
   定位字体路径，再由 vinext 转成 `/assets/_vinext_fonts/`。迁移后的缓存可离线复用，
   无需改写 node_modules。回归测试覆盖目录搬迁、重复执行、首次无缓存构建与最终 URL。
2. **新克隆缺构建源码。** 根 `.gitignore` 的 `**/build/` 误忽略了 Vite 显式导入的
   `task_queue_web/build/sites-vite-plugin.ts`。现在只放行该目录的 `.ts` 和 `.mjs`
   手写辅助源码，其他构建产物继续忽略。已用仅含发布候选文件的独立临时目录构建验证。
3. **服务注册入口残留旧测试结构。** `bootstrap_data_review_service` 的验证命令引用
   不存在的 `tests/test_data_review_manager.py`，且固定使用某个 Conda Python。
   已改为 `tests/unit/test_data_review_manager.py`、当前 `sys.executable` 和包模块启动；
   临时任务数据库测试验证注册、重复更新、命令导入及验证目标。本机线上服务仍由
   systemd 管理，审计没有向生产任务数据库注册第二个服务。
4. **论文图表引用错误目录。** 三份本地绘图脚本 `build_dataset_statistics_reference.py`、
   `update_paper_figure_labels.py`、`build_appendix_prompt_cards.py` 引用不存在的
   `论文/ICLR2027_SciDoc/`。已统一为实际 `论文/ICLR2027_ScienceDoc/`，并更新两份
   配套 README 和旧命令路径。论文目录继续按现有规则仅保留本地；未重新生成或覆盖图表。

## 检查范围与证据

- 源码、脚本、Notebook、文档、配置、部署单元、数据 JSON、模型配置、论文材料、
  前端构建缓存及 `sxz/` 均纳入静态检查；Git 对象、权重二进制、图片和压缩归档
  不作为可执行文本逐字扫描。没有执行可能写入 `sxz/` 的代码。
- 61 个 argparse 命令入口在仓库外 `/tmp` 使用包模块 `--help` 均成功。
- 96 个项目及论文 Python 文件编译检查通过，不写字节码缓存。
- 新增隔离目录回归测试复制源码和正式 QA，在禁用 site-packages 的 Python 中运行
  正式数据检查及四组件评测计划，确认不依赖本机包安装或旧目录。
- 全部 74 个符号链接均可解析；九个模型索引列出的全部 59 个权重分片存在。
- 默认模型及 processor 路径均落在 `/data/czj/SciDoc/models/`；Python editable 路径
  与安装元数据均指向新目录；正式评测计划不再含旧项目绝对路径。
- 已安装的 systemd 单元与仓库模板完全一致，服务工作目录和环境文件指向新目录。
  shell 配置、用户 systemd 配置及项目 `.env` 未发现旧项目路径；用户无 crontab。
- 审核库、队列库 SQLite `quick_check` 均为 `ok`；228 条审核快照引用全部存在，
  当前任务队列为空。
- 前端相对源码 import 全部能解析。临时独立目录仅复制 Git 发布候选文件，共用
  已安装依赖并复制字体缓存，构建与 8 项 Web 测试通过；该检查没有重新联网安装依赖。
- 本地详细日志在 `.migration-backup/path-audit-*`；它们不提交公开仓库。

## 最终验收

- 文档清单同步及检查通过；Python 全量 **425 项测试通过**。
- Web 构建及 **8 项测试通过**，独立临时目录构建同样通过。
- 正式 **2,200 条 QA / 712 份评测 PDF** 的内容、哈希、可读性和证据页范围检查通过。
  最终发布目录六个文件与迁移前 SHA-256 一致。
- 全量 PDF 资产检查为 `healthy`：**1,717 个规范资产**，未改写任何 JSON。
- 重启服务后，本地三个端点、VPS 隧道后端及公网入口全部为 200，`healthy: true`。
- 本地与公网分别逐个检查 **11 个字体 URL，全部返回 200**；渲染 HTML 不再含服务器磁盘路径。

## 保留的历史引用与限制

`sxz/` 和历史审计材料仍含 `/data/czj/pku/`，旧目录符号链接必须保留。
228 条可逆审核快照依赖这个别名；这些记录的历史身份没有被批量改写。

只读扫描发现 `sxz/` 有 485 处绝对路径字面量在当前文件系统中不存在；数据 JSON
中也有指向旧 `data/pdfs/papers/`、已归档 QA 或旧顶层模型目录的 provenance 字段。
逐项对照迁移前文件清单，这些缺失项在迁移前已不存在，未发现迁移丢失的目标。
它们是历史脚本或审计元数据的问题，不能通过把 `pku` 替换为 `SciDoc` 修复。
本次没有重建废弃目录、伪造旧数据、放宽断点协议或修改协作者目录。

正式数据使用当前 PDF 资产索引；未重新运行耗 GPU/外部 API 的整套推理与生成实验。
已完成的检查不能替代这些实验本身，也不表示全部历史脚本都已恢复可运行。
