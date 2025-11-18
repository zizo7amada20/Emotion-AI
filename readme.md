# Emotion_AI_RealTime_Project

## 📌 Overview
A **real-time emotion detection web application** built using **Streamlit**, **OpenCV**, **FER**, and **Mediapipe**.  
It captures live video from your webcam, instantly detects facial emotions (😊 happy, 😢 sad, 😡 angry, 😱 fear, 🤢 disgust, 😐 neutral), dynamically changes the background according to the dominant emotion, applies automatic lighting correction, and plots emotion intensity over time.

Perfect for interactive mood-based experiences, mental health monitoring demos, virtual backgrounds with feelings, or simply having fun watching AI react to your face in real time!

---

## ⚙ Requirements
streamlit==1.38.0  
opencv-python==4.10.0.84  
Pillow==10.4.0  
numpy==2.1.2  
fer==22.5.1  
mtcnn==0.1.1  
tensorflow==2.17.0  
mediapipe==0.10.18  
playsound==1.3.0  
requests==2.32.3  

### Setup
1. Install dependencies: `pip install -r requirements.txt`  
2. Create folders: `mkdir backgrounds alerts`  
3. Add your media:  
   - `backgrounds/` → happy.jpg, sad.jpg, angry.jpg, ...  
   - `alerts/` → happy.mp3, sad.mp3, angry.mp3, ...

---

## ▶ How to Run
1. `streamlit run app.py`  
2. Click **Start Camera**  
3. Allow webcam access  
4. Smile / frown / get angry → background + sound changes instantly

---

## 🛠 Problems Faced & Solutions
- **First run was slow** → models download only once  
- **Blurry background removal** → fixed using Mediapipe Selfie Segmentation  
- **Sound spam** → fixed with cooldown + threading  
- **Flickering in Streamlit** → fixed with `time.sleep(0.03)`  

💨 **Result:** Runs super smooth on any laptop.

---

## 💡 Benefits & Applications
- Learn **real-time Emotion AI** & **Computer Vision**  
- Mood-based virtual backgrounds  
- Mental health demo tools  
- Interactive games & installations
