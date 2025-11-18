import cv2
from fer import FER as FER_detector  # متوافق مع fer 25.x
import numpy as np
import mediapipe as mp
import playsound
import threading
import time
import os

# Initialize FER detector
detector = FER_detector(mtcnn=True)

# Initialize Mediapipe Selfie Segmentation
mp_selfie = mp.solutions.selfie_segmentation
segment = mp_selfie.SelfieSegmentation(model_selection=1)

# Last time an alert played
last_alert_time = 0

# Function to play voice alert in separate thread
def play_alert(path):
    if os.path.exists(path):
        threading.Thread(target=playsound.playsound, args=(path,), daemon=True).start()

# Auto light correction
def auto_light_correction(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# Replace background based on emotion
def replace_background(frame, emotion, backgrounds_path="backgrounds"):
    h, w = frame.shape[:2]
    bg_path = os.path.join(backgrounds_path, f"{emotion}.jpg")
    if not os.path.exists(bg_path):
        return frame
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (w, h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segment.process(rgb)
    mask = result.segmentation_mask
    mask = np.stack((mask, mask, mask), axis=-1)
    output = frame * mask + bg * (1 - mask)
    return output.astype(np.uint8)

# Analyze frame and apply features
def analyze_frame(frame, enable_light=True, enable_bg=True, enable_voice=True):
    global last_alert_time

    # Auto light
    if enable_light:
        frame = auto_light_correction(frame)

    # FER detection
    results = detector.detect_emotions(frame)
    emotion = "neutral"
    intensity = 0

    if results:
        emotions = results[0]["emotions"]
        emotion = max(emotions, key=emotions.get)
        intensity = round(emotions[emotion] * 100, 1)

    # Voice alert
    if enable_voice and intensity > 60:
        if time.time() - last_alert_time > 4:
            alert_path = f"alerts/{emotion}.mp3"
            play_alert(alert_path)
            last_alert_time = time.time()

    # Replace background
    if enable_bg and emotion:
        frame = replace_background(frame, emotion)

    return frame, emotion, intensity
