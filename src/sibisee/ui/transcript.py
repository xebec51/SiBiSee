from __future__ import annotations

from sibisee.domain.transcript import TranscriptBuilder


def get_transcript(st) -> TranscriptBuilder:
    if "transcript_builder" not in st.session_state:
        st.session_state.transcript_builder = TranscriptBuilder()
    return st.session_state.transcript_builder


def render_transcript_panel(st, transcript: TranscriptBuilder, last_token: str | None, confidence: float) -> None:
    st.subheader("Transcript")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Token terakhir", last_token or "-")
    col_b.metric("Confidence stabil", f"{confidence:.2f}" if confidence else "-")
    col_c.metric("Jumlah token", len(transcript.tokens))

    st.text_area("Hasil pengenalan", transcript.text, height=130)
    undo, clear, download = st.columns(3)
    if undo.button("Undo", use_container_width=True):
        transcript.undo()
        st.rerun()
    if clear.button("Clear", use_container_width=True):
        transcript.clear()
        st.rerun()
    download.download_button(
        "Download .txt",
        transcript.text.encode("utf-8"),
        file_name="sibisee-transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )
