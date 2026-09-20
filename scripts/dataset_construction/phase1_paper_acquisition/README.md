
## 第一阶段构建材料

本目录保留 SciDoc 的早期论文获取与基础 QA 处理流程：  
从 arXiv 自动下载论文 PDF（及源代码），经人工（或大模型）标注生成问答对（QA），再对数据进行格式转换、结构校验、类型统计、清洗过滤，输出过程性 Excel 统计报告。整个过程自动化程度高，便于后续用于模型训练、评估或数据分析。

---

## 文件列表与功能说明

本目录对应论文中的第一阶段“论文检索与基础数据构造”，文件按推荐执行顺序编号。输入、过程产物和统计结果应分别放入项目的 `data/` 子目录；不要在本目录生成运行数据。论文中的后续阶段（不可回答题、Reasoning、Cross-PDF、双模型复审和人工终审）由 `src/pku_qa/workflows/` 与 `data/qa/7.final_2200/` 的现行流程负责，本目录只保留早期生产链路的可复核记录。

项目共包含 6 个 Jupyter Notebook（`.ipynb`）脚本，按推荐执行顺序排列如下：

| 序号 | 文件名 | 功能描述 |
|------|--------|----------|
| 1    | `01_download_arxiv_papers.ipynb` | **批量下载 arXiv 论文**（无分类）<br>• 设置查询参数（日期范围、数量），调用 arXiv API 获取论文 ID 列表<br>• 将 ID 保存为文本文件<br>• 逐篇下载：优先下载 `tar.gz` 源码包，同时下载 PDF；若返回 `Content-Type` 为 PDF 则只下载 PDF，并分别统计数量<br>• 按 ID 创建子文件夹存放文件 |
| 2    | `02_download_by_category.ipynb` | **按学科分类下载**<br>• 基于自定义的“官方小类 → 大类/子类”映射表<br>• 为每个官方小类随机分配目标下载量（3 或 4 篇）<br>• 优先按日期范围搜索，不足时回退不限时间<br>• 全局去重，下载 tar+PDF，并为每篇文章生成元数据（编号、标题、作者、分类等），保存为 CSV 和 JSON 摘要 |
| 3    | `03_audit_pdf_pages.ipynb` | **PDF 页数统计**<br>• 扫描指定文件夹中的所有 PDF，提取页数<br>• 生成页数分布直方图<br>• 列出 ≤3 页和 ≥65 页的异常论文（用于筛选过短或过长文档） |
| 4    | `04_convert_excel_to_json.ipynb` | **Excel → JSON 转换**<br>• 将人工标注的 QA 数据（Excel 两列：序号 + JSON 字符串）转换为标准 JSON 对象文件<br>• 自动处理 LaTeX 命令转义，容错解析，输出格式化 JSON |
| 5    | `05_validate_and_clean_json.ipynb` | **JSON 质量检查、统计与历史清洗记录**（三合一）<br>• **① 结构验证**：检查论文与 QA 必填字段、分类、选项、模态、题型和证据页范围<br>• **② 类型统计**：记录早期选择题/简答题及数字答案分布<br>• **③ 历史清洗**：按早期规则生成基线并重新编号 QA1~QAn；最终发表集已在后续阶段统一为 short-answer-only，不能把本单元的历史规则当作最终 benchmark 规范 |
| 6    | `06_build_statistics_report.ipynb` | **生成 Excel 统计报告**<br>• 读取清洗后的 JSON，构建三大统计表：<br>　- *学科统计*：按一级学科 + 二级分类统计论文数量<br>　- *问题分类统计*：按一级学科 + 问题类别统计 QA 数量<br>　- *总体统计*：涵盖选择题/简答题、单模态/多模态、跨页/未跨页、字面/推断类分布<br>• 自动合并单元格、添加求和公式、设置边框与列宽，输出专业格式的过程性统计工作簿 |

---

## 推荐执行流程

1. **数据采集**（二选一）  
   - 如需按学科均匀采样 → 运行 `02_download_by_category.ipynb`  
   - 如只需简单批量下载 → 运行 `01_download_arxiv_papers.ipynb`  
   - 下载过程产物默认应保存在项目 `data/archive/` 或指定的数据工作目录，不要写入本目录。

2. **PDF 质量检查（可选）**  
   - 运行 `03_audit_pdf_pages.ipynb` 查看页数分布，剔除异常页数的论文。

3. **人工生成 QA 数据集**  
   - 利用下载的 PDF 文件，通过人工或大模型提示词生成问答对，整理为 Excel 文件（第一列为论文序号，第二列为包含该论文所有 QA 的 JSON 字符串）。

4. **数据导入与结构化**  
   - 运行 `04_convert_excel_to_json.ipynb`，将 Excel 转换为统一的 JSON 文件（如 `data/qa/1.base/work__single_pdf__raw_mixed__n6204.json`）。

5. **数据校验、统计与清洗**  
   - 运行 `05_validate_and_clean_json.ipynb`，依次执行：  
     - 结构验证 → 输出错误列表（可据此修正原始数据）  
     - 类型统计 → 了解 QA 构成  
     - 历史清洗 → 生成过程性基线；后续正式集合的 short-answer、精确 `Unanswerable` 和 `evidence_pages` 约束以当前 schema 和 manifest 为准。

6. **生成第一阶段统计报告**  
   - 运行 `06_build_statistics_report.ipynb`，输出过程性统计工作簿；正式统计以 `data/qa/7.final_2200/` 的当前文件和 manifest 为准。

---

## 依赖环境

- Python 3.8+
- 必需库：`arxiv`, `requests`, `numpy`, `pandas`, `openpyxl`, `PyPDF2`, `matplotlib`, `tqdm`, `json`, `re`, `collections`, `pathlib`

推荐使用虚拟环境安装：
```bash
pip install arxiv requests numpy pandas openpyxl PyPDF2 matplotlib tqdm

```
