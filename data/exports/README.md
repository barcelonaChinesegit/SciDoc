# 对外分发资源包

本目录存放准备上传网盘或交付的资源包及配套校验、恢复说明。
压缩包和交付目录仅保存在本机，不提交 Git。

当前 Google Drive 完整资源包位于 `google_drive/`：

- `ScienceDoc_PDFs_complete_20260920.zip`：全部 1,717 份 PDF（其中 712 份用于正式评测）、四个正式 QA 文件的原样副本及配套清单。
- `ScienceDoc_PDFs_complete_20260920.zip.sha256`：ZIP 的 SHA-256 校验文件。
- `README.md`：上传和恢复说明。
- `verification.json`：打包验证记录，`archive` 使用当前仓库相对路径。

上传前可从仓库根目录执行：

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
