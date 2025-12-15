# 👁️ Oculomics AI: Cardiovascular Risk & Biological Age Prediction

## 📌 Project Overview
This project utilizes **Deep Learning** and **Computer Vision** to analyze retinal fundus images. By examining the microvascular structure of the eye, the AI models predict:
1.  **Biological Age:** Estimating a patient's age to detect accelerated aging.
2.  **Cardiovascular Risk:** Predicting **Carotid Intima-Media Thickness (CIMT)**, a key biomarker for heart attack and stroke risk.

The project demonstrates the potential of non-invasive **Oculomics**—using the eye as a window to systemic health.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Deep Learning:** TensorFlow / Keras (CNN Architecture)
* **Computer Vision:** OpenCV (Green channel extraction, CLAHE)
* **Interface:** Streamlit (Web App)
* **Data Manipulation:** Pandas, NumPy

## 📂 Project Structure
```text
Eye_Project/
│
├── images/                  # Folder containing retinal scans (Left/Right eyes)
├── data_info.csv            # Dataset metadata (Age, CIMT labels)
├── requirements.txt         # List of dependencies
├── train_age_model.py       # Script to train the Age Predictor
├── train_heart_model.py     # Script to train the Heart Risk Predictor
├── app_age.py               # Streamlit App for Age Prediction
├── app_heart.py             # Streamlit App for Heart Risk
├── age_model.h5             # Saved Age Model (Generated after training)
├── heart_model.h5           # Saved Heart Model (Generated after training)
└── README.md                # Project Documentation


⚙️ Installation Guide
1. Clone the Repository (or download files)
Bash -> git clone [https://github.com/your-username/eye-project.git](https://github.com/your-username/eye-project.git)
cd eye-project

2. Create a Virtual Environment (Optional but Recommended)
Bash -> python -m venv venv
# Windows: -> venv\Scripts\activate
# Mac/Linux: -> source venv/bin/activate

3. Install Dependencies Run this command to install TensorFlow, OpenCV, and Streamlit automatically:
Bash -> pip install -r requirements.txt

🚀 How to Run

Step 1: Train the Models
If you haven't trained the AI yet, run these scripts. They will generate the .h5 model files.
# Train the Age Predictor
Bash -> python train_age_model.py

# Train the Heart Risk Predictor
Bash -> python train_heart_model.py

Step 2: Launch the Web App
You can run either the Age App or the Heart Risk App:
For Heart Attack Risk:
Bash -> streamlit run app_heart.py

For Biological Age:
Bash -> streamlit run app_age.py

📊 Results & Interpretation
1. Cardiovascular Risk Analysis (CIMT)

The model predicts the thickness of the Carotid Artery walls based on retinal vessel tortuosity.

Metric: Carotid Intima-Media Thickness (CIMT).

Threshold: > 0.90 mm indicates high cardiovascular risk.


[INSERT SCREENSHOT OF HEART APP HERE] (Replace this text with your screenshot, e.g., )

Interpretation: In the example above, the model analyzed the fundus image and predicted a CIMT of 1.2mm. Since this is above the 0.9mm threshold, the system flagged this patient as "High Risk", suggesting potential atherosclerosis or hypertension that requires clinical attention.

2. Biological Age Estimation

The model predicts the patient's age. Large discrepancies between "Predicted Age" and "Actual Age" (Retinal Age Gap) can indicate health issues.

[INSERT SCREENSHOT OF AGE APP HERE] (Replace this text with your screenshot)

Interpretation: The AI analyzed the vessel density and predicted an age of 68 years. If the patient is actually 50 years old, this "Age Gap" suggests their blood vessels are aging faster than normal, potentially due to smoking or diabetes.

📝 Dataset
Used the China Fundus (CIMT) Dataset containing 2,903 high-resolution fundus images with corresponding clinical labels for age, gender, and CIMT measurements.

🤝 Acknowledgements
Streamlit for the UI framework.

TensorFlow for the neural network backend.


---

### **Instructions for your Screenshots**

1.  **Run the App:** Run `streamlit run app_heart.py`.
2.  **Upload an Image:** Pick an image from your `images` folder (e.g., one where the filename in CSV has a thickness > 0.9).
3.  **Take a Screenshot:** Use the "Snipping Tool" (Windows) or `Cmd+Shift+4` (Mac) to take a picture of the web browser showing the "High Risk" result.
4.  **Save:** Save the image as `heart_result.png`.
5.  **Edit README:** Open the `README.md` file and drag-and-drop your image into the section where I wrote `[INSERT SCREENSHOT HERE]`.