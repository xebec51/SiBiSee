import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from twilio.rest import Client # Import library Twilio

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

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    model_path = 'models/best.pt' 
    model = load_model(model_path)
    # st.sidebar.success("Model berhasil dimuat!") # Optional: Matikan agar tidak memenuhi sidebar
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- FUNGSI DAPATKAN ICE SERVERS (TURN/STUN) ---
# Ini fungsi krusial untuk menembus Firewall Streamlit Cloud
@st.cache_data(ttl=600) # Cache 10 menit agar tidak boros kuota Twilio
def get_ice_servers():
    try:
        # Coba ambil credentials dari Secrets Streamlit
        account_sid = st.secrets["twilio"]["ACCOUNT_SID"]
        auth_token = st.secrets["twilio"]["AUTH_TOKEN"]
        
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        # Fallback jika gagal/belum setting secrets (Pakai Google STUN Gratisan)
        st.warning(f"Menggunakan STUN Google (Koneksi mungkin tidak stabil). Error Twilio: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- SIDEBAR (PENGATURAN) ---
st.sidebar.header("⚙️ Pengaturan Model")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
mode_select = st.sidebar.radio("Pilih Mode:", ["Live Kamera (Real-time)", "Upload Gambar"])

# --- FUNGSI CALLBACK VIDEO ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model(img, conf=conf_threshold)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- LAYOUT UTAMA ---
if mode_select == "Live Kamera (Real-time)":
    st.subheader("🔴 Deteksi Video Langsung")
    
    # Ambil konfigurasi server TURN/STUN
    ice_servers = get_ice_servers()
    rtc_configuration = RTCConfiguration({"iceServers": ice_servers})

    webrtc_streamer(
        key="sibisee-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration, # Masukkan konfigurasi Twilio disini
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif mode_select == "Upload Gambar":
    st.subheader("🖼️ Deteksi Gambar Statis")
    uploaded_file = st.file_uploader("Upload file JPG/PNG", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
        
        if st.button("🔍 Deteksi Sekarang", type="primary"):
            results = model.predict(image, conf=conf_threshold)
            res_plotted = results[0].plot()[:, :, ::-1]
            with col2:
                st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)
                
# --- FOOTER ---
st.divider()
st.caption("© 2025 SiBiSee Project | Powered by Streamlit & YOLOv8-CBAM")