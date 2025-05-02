import numpy as np
import streamlit as st
import cv2
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# Load ResNet50 for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extractor_resnet = Model(inputs=base_model.input, outputs=base_model.output)

# Load trained deepfake detection model
model = load_model('deepfake_resnet50.h5')

# Extract frames for processing
def extract_frames(video_path, num_frames=20):
    video = cv2.VideoCapture(video_path)
    frames, total_frames = [], int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = video.read()
        if ret:
            frame_resized = cv2.resize(frame, (224, 224))
            frames.append(preprocess_input(frame_resized))

    video.release()
    return np.array(frames)

# Predict deepfake probability
def predict_video(video_path):
    frames = extract_frames(video_path)
    frame_features = feature_extractor_resnet.predict(frames)
    frame_features_reshaped = frame_features.reshape(1, 20, 2048)
    additional_input = np.zeros((1, 20))  
    result = model.predict([frame_features_reshaped, additional_input])
    return "🔴 Fake Video" if result > 0.91 else "🟢 Real Video"

# Streamlit UI
st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🎭 Deepfake Detection System")
st.sidebar.header("Upload Video 🎥")

video_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if video_file:
    video_path = video_file.name
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.video(video_file)
    
    with col2:
        st.info("Click 'Analyze' to check for deepfake content after playing the video.")
        
        if st.button("🔍 Analyze Video"):
            with col1:
                st.video(video_file)  # Replaying video while processing
                time.sleep(5)  # Simulating playback duration before analysis
            
            with st.spinner("Analyzing... 🔄"):
                result = predict_video(video_path)
            st.success(f"*Result:* {result}")