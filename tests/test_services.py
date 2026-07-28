from __future__ import annotations

from pathlib import Path

from sibisee.services.gesture_guide import classify_label, discover_guide_items
from sibisee.services.ice_servers import DEFAULT_STUN_SERVERS, get_ice_servers


def test_twilio_fallback_without_credentials() -> None:
    assert get_ice_servers({}) == DEFAULT_STUN_SERVERS


def test_guide_discovery_and_category(tmp_path: Path) -> None:
    (tmp_path / "A.jpg").write_bytes(b"fake")
    (tmp_path / "Satu.png").write_bytes(b"fake")
    (tmp_path / "README.md").write_text("skip", encoding="utf-8")

    items = discover_guide_items(tmp_path)

    assert len(items) == 2
    assert classify_label("A") == "Alfabet"
    assert classify_label("Satu") == "Angka"
    assert classify_label("Makan") == "Kata"
