import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION ---
MODEL_FILENAME = 'age_model.h5' 

# --- SAFE MODEL LOADING (FIXED) ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_FILENAME):
        return None, f"File '{MODEL_FILENAME}' not found. Please run your training script first."
    
    try:
        # ---------------------------------------------------------
        # THE FIX IS HERE: compile=False
        # This tells Keras: "Don't worry about the training metrics, 
        # just load the weights for prediction."
        # ---------------------------------------------------------
        model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
        return model, None
    except Exception as e:
        return None, str(e)

# Initialize model
model, error_message = load_my_model()

# --- PREPROCESSING FUNCTION ---
def preprocess_image(image):
    IMG_SIZE = 128
    img_array = np.array(image)
    
    # Handle Grayscale vs RGB
    if len(img_array.shape) == 2:  # Grayscale
        img_green = img_array
    elif len(img_array.shape) == 3: # RGB
        # Resize first to ensure we work with correct dimensions
        img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_green = img_resized[:, :, 1] # Green Channel
    else:
        return None

    # Ensure resize (for grayscale path)
    if img_green.shape != (IMG_SIZE, IMG_SIZE):
        img_green = cv2.resize(img_green, (IMG_SIZE, IMG_SIZE))
        
    # Normalize
    img_normalized = img_green / 255.0
    
    # Reshape for Model: (1, 128, 128, 1)
    img_final = np.expand_dims(img_normalized, axis=0)
    img_final = np.expand_dims(img_final, axis=-1)
    
    return img_final

# --- THE WEBSITE UI ---
st.title("👁️ Retinal Age Predictor")

if model is None:
    st.error("⚠️ Model Loading Failed!")
    st.warning(error_message)
    st.stop()

st.write("Upload a fundus image to predict biological age.")

uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan', use_container_width=True)
    
    if st.button("Analyze Scan"):
        with st.spinner('Analyzing...'):
            processed_img = preprocess_image(image)
            
            if processed_img is not None:
                prediction = model.predict(processed_img)
                result = prediction[0][0]
                
                st.success("Analysis Complete")
                st.metric(label="Predicted Age", value=f"{result:.1f} Years")
            else:
                st.error("Could not process image.")