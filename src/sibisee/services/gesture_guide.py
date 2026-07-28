from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NUMBER_WORDS = {
    "Satu",
    "Dua",
    "Tiga",
    "Empat",
    "Lima",
    "Enam",
    "Tujuh",
    "Delapan",
    "Sembilan",
}


@dataclass(frozen=True)
class GuideItem:
    label: str
    category: str
    path: Path


def classify_label(label: str) -> str:
    if len(label) == 1 and label.isalpha():
        return "Alfabet"
    if label in NUMBER_WORDS:
        return "Angka"
    return "Kata"


def discover_guide_items(guide_dir: Path) -> tuple[GuideItem, ...]:
    if not guide_dir.exists():
        return ()
    items = [
        GuideItem(label=path.stem, category=classify_label(path.stem), path=path)
        for path in guide_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return tuple(sorted(items, key=lambda item: (item.category, item.label)))
