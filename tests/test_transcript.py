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
