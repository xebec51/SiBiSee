from __future__ import annotations

import datetime as _dt


def render_footer(st) -> None:
    st.divider()
    year = _dt.datetime.now().year
    st.caption(
        f"SiBiSee {year}. Pengenalan gestur SIBI terisolasi secara real-time. "
        "Hasil bersifat prediksi bantu dan dapat salah."
    )
