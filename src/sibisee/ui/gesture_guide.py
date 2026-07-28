from __future__ import annotations

from sibisee.services.gesture_guide import GuideItem


def render_guide(st, items: tuple[GuideItem, ...]) -> None:
    with st.expander("Panduan gestur SIBI"):
        if not items:
            st.info("Gambar panduan belum tersedia.")
            return
        query = st.text_input("Cari gestur", "")
        categories = ["Semua", "Alfabet", "Angka", "Kata"]
        category = st.segmented_control("Kategori", categories, default="Semua")

        filtered = [
            item
            for item in items
            if (category == "Semua" or item.category == category) and (not query or query.lower() in item.label.lower())
        ]
        cols = st.columns(4)
        for index, item in enumerate(filtered):
            with cols[index % 4]:
                st.image(str(item.path), caption=f"{item.label} - {item.category}", use_container_width=True)
