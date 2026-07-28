from __future__ import annotations

from sibisee.domain.transcript import TranscriptBuilder


def test_transcript_append_undo_clear() -> None:
    transcript = TranscriptBuilder()

    transcript.append("Saya")
    transcript.append("Makan")

    assert transcript.text == "Saya Makan"
    assert transcript.undo() == "Makan"
    assert transcript.text == "Saya"
    transcript.clear()
    assert transcript.text == ""


def test_transcript_snapshot_and_append_once() -> None:
    transcript = TranscriptBuilder()

    assert transcript.append_once("A") is True
    assert transcript.append_once("A") is False

    assert transcript.snapshot() == ("A",)
    assert transcript.last_token == "A"
