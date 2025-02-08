
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
import os

model_path = 'c:/Users/NARESH/OneDrive/Desktop/waste_classification_app/waste_classification_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model('waste_classification_model.h5')

def preprocess_image(image):
    """
    Preprocess the input image for prediction.
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(image):
    """
    Predict whether the image is Organic or Recyclable.
    """
    processed_img = preprocess_image(image)
    result = np.argmax(model.predict(processed_img))
    
    if result == 0:
        return "Organic Waste"
    elif result == 1:
        return "Recyclable Waste"