import streamlit as st
import cv2
import numpy as np
from skimage import feature
import pickle
from PIL import Image

# ── Load the trained model ──
# Opens the saved model file so we can use it to predict
with open('liveness_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ── LBP feature extraction function ──
# Converts any face image into 10 numbers the model understands
def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist

# ── App title and description ──
# This is what appears at the top of the web page
st.title("Face Liveness Detector")
st.write("Upload a face image to check if it is REAL or FAKE")

# ── File uploader ──
# Shows a button to upload an image
uploaded_file = st.file_uploader("Choose a face image", type=['jpg', 'jpeg', 'png'])

# ── Only run when an image is uploaded ──
if uploaded_file is not None:

    # Convert uploaded file to an image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Show the uploaded image on screen
    st.image(image, caption="Uploaded image", width=300)

    # Extract LBP features from the image
    features = extract_lbp(img_array)

    # Run the model prediction
    prediction = model.predict([features])[0]
    confidence = model.predict_proba([features])[0]

    # Show the result
    if prediction == 1:
        st.success(f"REAL face — confidence {round(confidence[1]*100, 1)}%")
    else:
        st.error(f"FAKE face — confidence {round(confidence[0]*100, 1)}%")