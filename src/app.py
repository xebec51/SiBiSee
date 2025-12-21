import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

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
""")

# --- SIDEBAR (PENGATURAN) ---
st.sidebar.header("⚙️ Pengaturan Model")

# Load Model (Cache agar tidak reload terus)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
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
    value=0.40, # Naikkan sedikit agar kotak tidak flickering
    step=0.05
)

# --- PILIHAN MODE ---
mode_select = st.sidebar.radio(
    "Pilih Mode:",
    ["Live Kamera (Real-time)", "Upload Gambar"]
)

# --- FUNGSI CALLBACK UNTUK VIDEO STREAMING ---
def video_frame_callback(frame):
    # Konversi frame WebRTC (av.VideoFrame) ke format OpenCV (numpy array)
    img = frame.to_ndarray(format="bgr24")

    # Lakukan prediksi YOLO pada frame tersebut
    # stream=True membuat inferensi lebih cepat untuk video
    results = model(img, conf=conf_threshold)

    # Gambar kotak bounding box ke frame
    annotated_frame = results[0].plot()

    # Kembalikan frame yang sudah digambar ke browser
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- LAYOUT UTAMA ---
if mode_select == "Live Kamera (Real-time)":
    st.subheader("🔴 Deteksi Video Langsung")
    st.write("Izinkan akses kamera browser Anda. Deteksi akan berjalan otomatis.")

    # Konfigurasi STUN Server (Penting untuk Deploy Cloud agar tidak blank)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Jalankan Streamer WebRTC
    webrtc_streamer(
        key="sibisee-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif mode_select == "Upload Gambar":
    st.subheader("🖼️ Deteksi Gambar Statis")
    uploaded_file = st.file_uploader("Upload file JPG/PNG", type=['jpg', 'png', 'jpeg'])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
            
        if st.button("🔍 Deteksi Sekarang", type="primary"):
            # Prediksi
            results = model.predict(image, conf=conf_threshold)
            res_plotted = results[0].plot()[:, :, ::-1] # BGR ke RGB
            
            with col2:
                st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)
                
                # Tampilkan teks hasil
                names = model.names
                detected_cls = results[0].boxes.cls.cpu().numpy()
                if len(detected_cls) > 0:
                    unique_cls = set(detected_cls) # Hapus duplikat agar rapi
                    st.success(f"Terdeteksi: {', '.join([names[int(c)] for c in unique_cls])}")
                else:
                    st.warning("Tidak ada gestur terdeteksi.")

# --- FOOTER ---
st.divider()
st.caption("© 2025 SiBiSee Project | Powered by Streamlit & YOLOv8-CBAM")