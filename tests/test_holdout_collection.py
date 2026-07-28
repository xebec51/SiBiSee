from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_holdout import HoldoutSession, collect_holdout, validate_session


def test_collect_holdout_dry_run_writes_metadata_files(tmp_path: Path) -> None:
    session = HoldoutSession(
        participant_id="P000",
        session_id="S000",
        device_label="dry-run",
        background="plain",
        lighting="indoor",
        distance="desk",
    )

    output_dir = collect_holdout(tmp_path / "holdout", ["A"], session, dry_run=True)

    metadata_rows = list(csv.DictReader((output_dir / "metadata.csv").open(encoding="utf-8")))
    summary = json.loads((output_dir / "session_summary.json").read_text(encoding="utf-8"))

    assert metadata_rows == []
    assert summary["session"]["participant_id"] == "P000"
    assert summary["class_names"] == ["A"]


def test_holdout_session_rejects_empty_required_identity_fields() -> None:
    session = HoldoutSession(
        participant_id="",
        session_id="S000",
        device_label="dry-run",
        background="plain",
        lighting="indoor",
        distance="desk",
    )

    try:
        validate_session(session)
    except ValueError as exc:
        assert "participant_id" in str(exc)
    else:
        raise AssertionError("validate_session should reject empty participant_id")
