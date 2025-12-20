import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="SiBiSee - SIBI Detection",
    page_icon="👋",
    layout="wide"
)

# --- JUDUL & DESKRIPSI ---
st.title("👋 SiBiSee: Deteksi SIBI Real-time")
st.markdown("""
Aplikasi ini menggunakan **YOLOv8 + CBAM** untuk mendeteksi Sistem Isyarat Bahasa Indonesia (SIBI).
Dibuat oleh: **Muh. Rinaldi Ruslan**
""")

# --- SIDEBAR (PENGATURAN) ---
st.sidebar.header("⚙️ Pengaturan Model")

# Load Model (Cache agar tidak reload terus)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Pastikan path model sesuai dengan lokasi file .pt Anda di folder models/
try:
    # Ganti 'best.pt' dengan nama file model Anda yang sebenarnya
    model_path = 'models/best.pt' 
    model = load_model(model_path)
    st.sidebar.success("Model berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"Gagal memuat model: {e}")
    st.stop()

# Slider Confidence Threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

# --- PILIHAN SUMBER GAMBAR ---
source_radio = st.sidebar.radio(
    "Pilih Sumber Gambar:",
    ["Upload Gambar", "Gunakan Kamera"]
)

# --- FUNGSI DETEKSI ---
def detect_objects(image, conf):
    # Lakukan prediksi
    results = model.predict(image, conf=conf)
    
    # Plot hasil (menggambar kotak di gambar)
    # [:, :, ::-1] mengubah BGR ke RGB agar warna benar di web
    res_plotted = results[0].plot()[:, :, ::-1]
    
    return res_plotted, results

# --- LOGIKA TAMPILAN UTAMA ---
col1, col2 = st.columns(2)

input_image = None

with col1:
    st.subheader("1. Input Citra")
    
    if source_radio == "Upload Gambar":
        uploaded_file = st.file_uploader("Upload file JPG/PNG", type=['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Gambar yang diupload", use_container_width=True)
            
    elif source_radio == "Gunakan Kamera":
        camera_file = st.camera_input("Ambil foto gestur tangan")
        if camera_file is not None:
            input_image = Image.open(camera_file)
            # Tampilan kamera sudah otomatis muncul di widget

with col2:
    st.subheader("2. Hasil Deteksi")
    
    if input_image is not None:
        # Tombol Deteksi
        if st.button("🔍 Deteksi SIBI", type="primary"):
            with st.spinner('Sedang memproses...'):
                # Proses Deteksi
                result_img, result_data = detect_objects(input_image, conf_threshold)
                
                # Tampilkan Gambar Hasil
                st.image(result_img, caption="Hasil Deteksi YOLOv8-CBAM", use_container_width=True)
                
                # Tampilkan Detail Kelas (Opsional)
                st.success("Deteksi Selesai!")
                
                # Menampilkan teks hasil prediksi
                names = model.names
                detected_cls = result_data[0].boxes.cls.cpu().numpy()
                if len(detected_cls) > 0:
                    st.write("Terdeteksi:")
                    for cls_id in detected_cls:
                        st.info(f"👉 Huruf: **{names[int(cls_id)]}**")
                else:
                    st.warning("Tidak ada gestur yang terdeteksi.")
    else:
        st.info("Silakan upload gambar atau ambil foto untuk memulai.")

# --- FOOTER ---
st.divider()
st.caption("© 2025 SiBiSee Project. Dikembangkan dengan Streamlit & YOLOv8.")