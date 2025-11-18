import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time
import requests

# -------------------- Emotion Detector --------------------
try:
    from fer import FER
    detector = FER(mtcnn=True)
except:
    detector = None

# -------------------- Backgrounds --------------------
happy_bg = "https://i.imgur.com/OQ9QG5G.jpeg"
sad_bg = "https://i.imgur.com/n6C2GQjj.jpeg"
angry_bg = "https://i.imgur.com/iQ0SQ0r.jpeg"

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="Emotion AI", layout="wide")

# -------------------- CSS --------------------
st.markdown("""
<style>
.stApp {
    height: 100vh;
    background-size: 350% 350%;
    animation: gradientBG 12s ease-in-out infinite;
}
@keyframes gradientBG {
    0% { background: linear-gradient(135deg, #dfe4ea, #ced6e0); }
    25% { background: linear-gradient(135deg, #ced6e0, #e0e5ec); }
    50% { background: linear-gradient(135deg, #e0e5ec, #dfe4ea); }
    75% { background: linear-gradient(135deg, #dfe4ea, #ced6e0); }
    100% { background: linear-gradient(135deg, #ced6e0, #e0e5ec); }
}
.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #333;
    margin-top: -20px;
    text-shadow: 0px 0px 15px rgba(0,0,0,0.2);
}
.card {
    backdrop-filter: blur(16px);
    background: rgba(255,255,255,0.25);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
div.stButton > button {
    background: linear-gradient(90deg, #6c757d, #adb5bd);
    color: white;
    padding: 0.75rem 1.3rem;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 700;
    border: none;
    box-shadow: 0px 0px 15px rgba(108,117,125,0.5);
}
div.stButton > button:hover {
    transform: scale(1.04);
}
.stop-btn button {
    background: linear-gradient(90deg, #495057, #6c757d) !important;
    box-shadow: 0px 0px 15px rgba(73,80,87,0.5) !important;
}
.camera-frame {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# -------------------- Title --------------------
st.markdown('<div class="title">Emotion AI - Real Time Detection</div>', unsafe_allow_html=True)

# -------------------- Session Init --------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "log" not in st.session_state:
    st.session_state.log = []
if "graph_data" not in st.session_state:
    st.session_state.graph_data = []

# -------------------- Layout --------------------
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Start Camera"):
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
        st.session_state.run = True

with col2:
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    if st.button("Stop Camera"):
        st.session_state.run = False
        try: st.session_state.cap.release()
        except: pass
        st.session_state.cap = None
    st.markdown('</div>', unsafe_allow_html=True)

frame_placeholder = st.empty()
chart_placeholder = st.empty()

# -------------------- Background Replace Function --------------------
def replace_background(frame, bg_url):
    try:
        bg = Image.open(requests.get(bg_url, stream=True).raw).resize((frame.shape[1], frame.shape[0]))
        bg = np.array(bg)
        mask = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mask = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.GaussianBlur(mask, (25,25), 0)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255
        return frame * mask + bg * (1 - mask)
    except:
        return frame

# -------------------- Loop --------------------
while st.session_state.run and st.session_state.cap is not None:

    ret, frame = st.session_state.cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.convertScaleAbs(frame, alpha=1.18, beta=18)

    emotion_text = ""
    score = 0
    bg_url = None

    if detector is not None:
        try:
            result = detector.top_emotion(frame)
            if result and result[0]:
                emotion, score = result
                emotion_text = f"{emotion} ({score:.2f})"
                
                # Choose background
                if emotion == "happy":
                    bg_url = happy_bg
                elif emotion == "sad":
                    bg_url = sad_bg
                elif emotion == "angry":
                    bg_url = angry_bg

                # Log CSV
                st.session_state.log.append([time.time(), emotion, score])

                # Add graph point
                st.session_state.graph_data.append(score)

                # Trigger action
                if emotion == "happy":
                    st.image("https://i.imgur.com/tSmNXdP.jpeg", width=250)
                elif emotion == "sad":
                    pass  # الأغنية اتشالت
                elif emotion == "angry":
                    pass  # already has special red bg
        except:
            pass

    # Background effect
    if bg_url:
        frame = replace_background(frame, bg_url)

    # Draw text
    cv2.putText(frame, emotion_text, (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

    # Show frame
    frame_placeholder.image(frame, channels="RGB", use_column_width=True)

    # Update graph
    if len(st.session_state.graph_data) > 2:
        chart_placeholder.line_chart(st.session_state.graph_data)

    time.sleep(0.03)
    st.experimental_rerun()
