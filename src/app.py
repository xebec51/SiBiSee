import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from twilio.rest import Client
import datetime
from cryptography.fernet import Fernet
import os

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
Gunakan kamera atau upload gambar untuk memulai deteksi.
""")

# --- [BARU] PANDUAN GESTURE (EXPANDER) ---
with st.expander("ℹ️ Klik di sini untuk melihat Panduan Gesture SIBI (Contoh A-Z)"):
    st.write("Gunakan gambar di bawah ini sebagai acuan untuk membentuk gerakan tangan Anda:")
    # Pastikan file 'panduan_sibi.jpg' sudah ada di folder 'assets/' Anda.
    # Jika nama file berbeda, silakan ubah path di bawah ini.
    try:
        st.image("assets/panduan_sibi.jpg", caption="Contoh Alfabet SIBI", use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ File gambar panduan ('assets/panduan_sibi.jpg') belum ditemukan. Silakan tambahkan file gambar ke folder assets.")


# --- FUNGSI DEKRIPSI & LOAD MODEL (SECURE) ---
@st.cache_resource
def load_model():
    encrypted_path = 'models/best.pt.enc'
    decrypted_path = 'temp_model.pt'
    
    if not os.path.exists(encrypted_path):
        st.error("File model terenkripsi (best.pt.enc) tidak ditemukan di server!")
        st.stop()
        
    try:
        if "model_security" in st.secrets:
            key = st.secrets["model_security"]["ENCRYPTION_KEY"]
        else:
            st.error("Kunci enkripsi tidak ditemukan di Secrets!")
            st.stop()
            
        fernet = Fernet(key)

        with open(encrypted_path, "rb") as file:
            encrypted_data = file.read()
            
        decrypted_data = fernet.decrypt(encrypted_data)

        with open(decrypted_path, "wb") as file:
            file.write(decrypted_data)
        
        model = YOLO(decrypted_path)
        
        return model

    except Exception as e:
        st.error(f"Gagal mendekripsi model. Pastikan kunci di Secrets benar. Error: {e}")
        st.stop()

model = load_model()

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
    
    img_source = st.radio(
        "Pilih Sumber Gambar:", 
        ("Upload File", "Ambil Foto (Kamera)"), 
        horizontal=True
    )
    
    input_image = None
    
    if img_source == "Upload File":
        uploaded_file = st.file_uploader("Upload file JPG/PNG", type=['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            
    elif img_source == "Ambil Foto (Kamera)":
        camera_file = st.camera_input("Ambil foto gestur tangan")
        if camera_file is not None:
            input_image = Image.open(camera_file)
    
    if input_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="Gambar Asli", use_container_width=True)
        
        if st.button("🔍 Deteksi Sekarang", type="primary"):
            with st.spinner('Sedang memproses...'):
                results = model.predict(input_image, conf=conf_threshold)
                res_plotted = results[0].plot()[:, :, ::-1]
                
                with col2:
                    st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)
                    names = model.names
                    detected_cls = results[0].boxes.cls.cpu().numpy()
                    
                    if len(detected_cls) > 0:
                        unique_cls = set(detected_cls)
                        st.success("Terdeteksi:")
                        for cls_id in unique_cls:
                            st.info(f"👉 **{names[int(cls_id)]}**")
                    else:
                        st.warning("Tidak ada gestur yang terdeteksi.")

# --- FOOTER (SOSIAL MEDIA & COPYRIGHT) ---
st.divider()

st.markdown("<p style='text-align: center; color: gray;'>Connect with Developer:</p>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.link_button("GitHub", "https://github.com/xebec51", use_container_width=True)
with c2:
    st.link_button("LinkedIn", "https://www.linkedin.com/in/rinaldiruslan", use_container_width=True)
with c3:
    st.link_button("Instagram", "https://instagram.com/rinaldiruslan", use_container_width=True)
with c4:
    st.link_button("TikTok", "https://tiktok.com/@rinaldiruslan", use_container_width=True)
with c5:
    st.link_button("Facebook", "https://web.facebook.com/rinaldi.naldi.5220", use_container_width=True)

current_year = datetime.datetime.now().year
st.markdown(f"""
<div style="text-align: center; margin-top: 15px; font-size: 12px; color: gray;">
    © {current_year} Naldi. All rights reserved. <br>
    Built with <b>Streamlit</b> & <b>YOLOv8-CBAM</b>
</div>
""", unsafe_allow_html=True)