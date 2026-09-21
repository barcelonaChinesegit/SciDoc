# 对外分发资源包

本目录存放对外分发资源包的本机打包副本及配套校验、恢复说明，不提交压缩包到 Git。
PDF 下载入口为 [公开 Google Drive 文件夹](https://drive.google.com/drive/folders/1J7l5HHPlKyjjegxZ2c0_plOnk9CdMVbY)。
下载解压后，保留原始文件名并将 PDF 平铺到仓库的 `data/pdfs/`；
校验命令见 [PDF 分发说明](../../docs/releases/PDF_DISTRIBUTION.md#download-and-validate)。

当前完整资源包的本机副本位于 `google_drive/`：

- `ScienceDoc_PDFs_complete_20260920.zip`：全部 1,717 份 PDF（其中 712 份用于正式评测）、四个正式 QA 文件的原样副本及配套清单。
- `ScienceDoc_PDFs_complete_20260920.zip.sha256`：ZIP 的 SHA-256 校验文件。
- `README.md`：上传和恢复说明。
- `verification.json`：打包验证记录，`archive` 使用当前仓库相对路径。

本机打包副本可从仓库根目录执行以下命令校验：

```bash
cd data/exports/google_drive
sha256sum -c ScienceDoc_PDFs_complete_20260920.zip.sha256
```

源 PDF 保留在 `data/pdfs/`，正式 QA 保留在 `data/qa/7.final_2200/`；
评测运行输出继续写入 `data/results/`。

## 评分回放材料

上述 PDF/QA 资源包不包含原实验五组件结果、v4 Judge 缓存或原始 Table 3 学科清单。
[评测 Quick Start](../../evaluation/README.md#quick-start) 列出了评分回放的独立材料；
缓存回放本身不需要 PDF。新 PDF 推理和当前发布集预检才需要部署这里的 PDF 包。
