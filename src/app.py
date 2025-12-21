import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from twilio.rest import Client

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
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- FUNGSI ICE SERVERS (TWILIO/STUN) ---
@st.cache_data(ttl=600)
def get_ice_servers():
    try:
        account_sid = st.secrets["twilio"]["ACCOUNT_SID"]
        auth_token = st.secrets["twilio"]["AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning(f"Menggunakan STUN Google (Koneksi mungkin tidak stabil). Error Twilio: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- SIDEBAR (PENGATURAN) ---
st.sidebar.header("⚙️ Pengaturan Model")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)

# Update Pilihan Mode: Ubah nama agar lebih jelas
mode_select = st.sidebar.radio(
    "Pilih Mode:", 
    ["Live Kamera (Real-time)", "Gambar Statis (Foto/Upload)"]
)

# --- FUNGSI CALLBACK VIDEO LIVE ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model(img, conf=conf_threshold)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- LOGIKA UTAMA ---

# 1. MODE LIVE VIDEO STREAMING
if mode_select == "Live Kamera (Real-time)":
    st.subheader("🔴 Deteksi Video Langsung")
    ice_servers = get_ice_servers()
    rtc_configuration = RTCConfiguration({"iceServers": ice_servers})

    webrtc_streamer(
        key="sibisee-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# 2. MODE GAMBAR STATIS (UPLOAD ATAU SNAPSHOT)
elif mode_select == "Gambar Statis (Foto/Upload)":
    st.subheader("🖼️ Deteksi Gambar Statis")
    
    # Pilihan Sub-Metode: Upload File atau Ambil Foto
    img_source = st.radio(
        "Pilih Sumber Gambar:", 
        ("Upload File", "Ambil Foto (Kamera)"), 
        horizontal=True
    )
    
    input_image = None
    
    # Logika Input
    if img_source == "Upload File":
        uploaded_file = st.file_uploader("Upload file JPG/PNG", type=['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            
    elif img_source == "Ambil Foto (Kamera)":
        camera_file = st.camera_input("Ambil foto gestur tangan")
        if camera_file is not None:
            input_image = Image.open(camera_file)
    
    # Proses Deteksi (Sama untuk kedua sumber)
    if input_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(input_image, caption="Gambar Asli", use_container_width=True)
        
        # Tombol Eksekusi
        if st.button("🔍 Deteksi Sekarang", type="primary"):
            with st.spinner('Sedang memproses...'):
                # Prediksi YOLO
                results = model.predict(input_image, conf=conf_threshold)
                
                # Plotting
                res_plotted = results[0].plot()[:, :, ::-1] # Konversi BGR ke RGB
                
                with col2:
                    st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)
                    
                    # Tampilkan Teks Kelas
                    names = model.names
                    detected_cls = results[0].boxes.cls.cpu().numpy()
                    
                    if len(detected_cls) > 0:
                        unique_cls = set(detected_cls)
                        st.success("Terdeteksi:")
                        for cls_id in unique_cls:
                            st.info(f"👉 **{names[int(cls_id)]}**")
                    else:
                        st.warning("Tidak ada gestur yang terdeteksi.")

# --- FOOTER ---
st.divider()
st.caption("© 2025 SiBiSee Project | Powered by Streamlit & YOLOv8-CBAM")