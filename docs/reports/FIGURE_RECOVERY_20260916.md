# 论文插图文件恢复（2026-09-16）

> 历史记录：文中旧路径与当时测试结果保留用于追溯。当前 Web 源码在
> `tools/internal/experiment_console/web/`；当前论文协议和复现结论见
> [2026-09-20 评测审计](official_evaluation/FINAL_REPORT.md)。

恢复范围为 `论文/写作材料/图/`，共 22 个原文件：两个 Python 模块、说明、提示词、审核截图素材、七张成品图及十张案例截图。

`/论文/` 被 Git 忽略，分支历史、暂存区和对象检查未找到这些文件的可用原版。成品图和素材来自当天交付副本；原始生成器通过 VS Code 本地历史与保存的后续修改记录恢复，正文模块和说明来自保存的原文。前 12 项全部与删除前 `source_copy_manifest.json` 的 SHA-256 一致，未以交付包中删去 Word 功能的脚本替代原版。

十张案例截图按原恢复脚本的章节与图片关系，从 `/tmp/附录部分中文草稿.before_layout.docx` 提取原始图片字节，未重新渲染。回收站中 9 月 11 日的过时文件未恢复。此次没有恢复 Word 稿件或其他论文目录。

## 恢复文件校验值

| 原路径 | SHA-256 |
| --- | --- |
| `论文/写作材料/图/图表生成代码/build_dataset_figures.py` | `99ba65150978f128fbe9a99c05ad62dcb0b331c2eb424b7e246e93f9b0d6e393` |
| `论文/写作材料/图/图表生成代码/dataset_section_content.py` | `a3aca8605c5a3a3887d6e45dc3e74a94b06d9ce9ed18c1d47b5d85492c442bed` |
| `论文/写作材料/图/图表生成代码/README.md` | `05d147bca58348549ecd940dc12f995a96006839fda9ff8ef9e916bc7ac08fc2` |
| `论文/写作材料/图/图表生成代码/assets/manual_review_source.png` | `b9f0d97f77844160fd0614307f99b77b877d6ae131a9d0eadfe7212e46527ca1` |
| `论文/写作材料/图/图表生成代码/prompts/fig_3_3_qa_composition_gpt_image_2_5.txt` | `b6322dba6433652c790ad4545c9b3c9c0995c80a5d416e0b3ee51390b45eb5da` |
| `论文/写作材料/图/第三部分_数据集/fig_3_0_scidoc_overview.png` | `ca58f7ebdb371495900cc4261f6148b92ac91b048b6638af3426347860a08244` |
| `论文/写作材料/图/第三部分_数据集/fig_3_1_primary_domain_distribution.png` | `f44b1d8a9d7bdf069701a5aae52e87b224363d549e4f4ef397c4c4c428c6fde2` |
| `论文/写作材料/图/第三部分_数据集/fig_3_2_sci_doc_pipeline.png` | `adaca10ce93311cc5e788064768edd428b6b87133efceb3c50a89170398a3146` |
| `论文/写作材料/图/第三部分_数据集/fig_3_3_qa_composition.png` | `61924744eb5212540491891f1aefcd145a216cc9130bc9860aae98e37c49123a` |
| `论文/写作材料/图/附录/fig_appendix_l_1_evidence_prf_example.png` | `e3b5a5da0b6ad12278d1b5c0c12756ed97f1e8941cb6a5e5ddc69bedea49a2a6` |
| `论文/写作材料/图/附录/fig_appendix_h_1b_manual_review_evidence.png` | `8231278ddb541ef4381c8d8c48307d0070fe2562b0bbc4af3ec5e9942a1f883d` |
| `论文/写作材料/图/附录/fig_appendix_h_1a_manual_review_overview.png` | `07f04735e36d717f9179e7ee98b99e10e8608eb131bb649c401b57b9c0afd210` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_1_1.png` | `f7b12dbaaeb51985e7f24a3596f7eb37d61a77b968708f21172544fd39232794` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_1_2.png` | `582fb02b202d8f711f7856c81764879b8c0ed12f9ca50911b52f99967a104af4` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_2_1.png` | `913e4c5e5778f8c88326ddb33c59e7ba941c55e24f2e8091d61fc13962ecd6f3` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_2_2.png` | `6afc5def1fbaa0cc3b10c22d994a215ed30d3a29db4651f81238ed01d5dbeeed` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_3_1.png` | `0e0ce3eea073c655d81b24b6852809947652bf6684390f68a552bab0a6af8565` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_3_2.png` | `129bf3b1117e102cb0b3b2bccdd716632bdbebc789a85d9cfee3b385ccc4272e` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_4_1.png` | `e67bf63e243b934ccd7054af13db6184db2811a397e64a7c117b7ea004354151` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_4_2.png` | `080d4972d3235a40ff960b753876f6cd6a3b0045a8ec09f6b5e614e70e9afadc` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_5_1.png` | `020a0d24d26b8cadac2d4d9476b3fccc12ff7a144d568c113ea068d812e69c4c` |
| `论文/写作材料/图/附录/五个案例/fig_appendix_l_case_5_2.png` | `a4e787d7aa7a73635b3d207fd16b045b774c9aa6059f4b0944ff2539ebd52188` |
