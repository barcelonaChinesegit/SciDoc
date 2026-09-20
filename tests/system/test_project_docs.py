from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]


def test_project_inventory_is_current() -> None:
    inventory = ROOT / "docs/current/PROJECT_INVENTORY.md"
    assert inventory.is_file()


def test_reader_documentation_links_and_images_exist() -> None:
    documents = [ROOT / "README.md", *(ROOT / "docs/current").glob("*.md")]
    for document in documents:
        source = document.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", source):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            path = document.parent / unquote(parsed.path)
            assert path.exists(), f"Broken documentation link in {document}: {target}"
