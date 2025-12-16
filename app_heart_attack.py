import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os


# --- CONFIGURATION ---
MODEL_FILENAME = 'heart_model.h5' 

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_FILENAME):
        return None, f"File '{MODEL_FILENAME}' not found. Please run 'train_heart_model.py' first."
    
    try:
        # Load with compile=False to avoid "mse" errors
        model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
        return model, None
    except Exception as e:
        return None, str(e)

model, error_message = load_my_model()

# --- PREPROCESSING ---
def preprocess_image(image):
    IMG_SIZE = 128
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_green = img_array
    elif len(img_array.shape) == 3:
        img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_green = img_resized[:, :, 1]
    else:
        return None

    if img_green.shape != (IMG_SIZE, IMG_SIZE):
        img_green = cv2.resize(img_green, (IMG_SIZE, IMG_SIZE))
        
    img_normalized = img_green / 255.0
    img_final = np.expand_dims(img_normalized, axis=0)
    img_final = np.expand_dims(img_final, axis=-1)
    
    return img_final

# --- UI ---
st.title("❤️ Cardiovascular Risk AI")
st.write("Analyzes retinal vessels to predict Carotid Intima-Media Thickness (CIMT).")

if model is None:
    st.error("⚠️ Model Loading Failed!")
    st.warning(error_message)
    st.stop()

uploaded_file = st.file_uploader("Upload Retinal Scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Patient Scan', use_container_width=True)
    
    if st.button("Assess Risk"):
        with st.spinner('Analyzing vessel density and tortuosity...'):
            processed_img = preprocess_image(image)
            
            if processed_img is not None:
                prediction = model.predict(processed_img)
                cimt_value = prediction[0][0]
                
                # --- MEDICAL LOGIC ---
                st.write("---")
                st.metric(label="Predicted Artery Thickness (CIMT)", value=f"{cimt_value:.3f} mm")
                
                # Risk Threshold: 0.9 mm is the clinical cutoff
                if cimt_value >= 0.9:
                    st.error("⚠️ HIGH RISK DETECTED")
                    st.write("**Interpretation:** The AI detected thickening of the arterial walls (>0.9mm). This is a strong biomarker for atherosclerosis, hypertension, and potential heart attack risk.")
                else:
                    st.success("✅ LOW RISK")
                    st.write("**Interpretation:** Arterial thickness is within the normal range (<0.9mm).")
            else:
                st.error("Image error.")