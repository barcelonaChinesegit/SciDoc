#!/usr/bin/env python3
"""Generate the human-readable catalog for every QA-shaped JSON file."""

from __future__ import annotations

from pathlib import Path

from pku_qa.services.review.data_review_manager import DatasetRegistry


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/current/DATASET_CATALOG.md"


def main() -> None:
    rows = DatasetRegistry().list()
    lines = [
        "# QA 数据集目录",
        "",
        "本目录由 `scripts/generate_dataset_catalog.py` 根据当前 `data/qa/**/*.json` 自动生成。",
        "网页“数据集与人工校验”会同步展示同一批可识别为 QA 的文件；位于 `data/qa/` 的模型输出、",
        "断点和汇总材料若仍含 QA 结构，也会列出但明确标为只读过程产物。`data/web/` 中的账户、",
        "审核事件、撤销快照和任务状态不参与数据集扫描。总数、哈希和更新时间",
        "以网页/API 返回为准。不要把本表中的每个 JSON 都当作正式发表输入。",
        "",
        f"当前识别到 **{len(rows)}** 个 QA-shaped JSON，各文件 QA 记录数相加为 **{sum(row.qas for row in rows)}**（含重复谱系和镜像，非独立题数）。",
        "最终 benchmark 仅为 `7.final_2200/` 下四个 QA 文件的 **2,200** 条。其余条目不增加发布规模。",
        "",
        "| 文件 ID | 相对路径 | QA / PDF | 状态与类别 | 来源/获得方式 | 用途 | 权限 |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        profile = row.profile
        path = profile.get("relative_path", row.path)
        source = profile.get("source_summary", profile.get("lineage", ""))
        use = profile.get("intended_use", profile.get("description", ""))
        permission = "可人工修改" if profile.get("reviewable", True) else "只读：" + str(profile.get("read_only_reason", "过程产物"))
        def cell(value: object) -> str:
            return str(value).replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| `{}` | `{}` | {} / {} | {}；{} | {} | {} | {} |".format(
                cell(row.id), cell(path), row.qas, row.papers,
                cell(profile.get("status", "")), cell(profile.get("artifact_class", "")),
                cell(source), cell(use), cell(permission),
            )
        )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
