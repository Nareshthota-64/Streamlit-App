import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import predict_image
from streamlit_extras.let_it_rain import rain

# Title of the app
st.set_page_config(page_title="Waste Classification App", page_icon="🌫", layout="centered")
st.title("🌫 Waste Classification using CNN")

# Subtitle and description
st.markdown(
    """
    ### Welcome to the Waste Classification App!
    This project aims to develop a Convolutional Neural Network (CNN) model for accurate image-based classification of waste.
    By effectively categorizing waste, the model contributes to improved waste management practices, enhanced resource recovery,
    and ultimately a more sustainable environment.

    **Upload an image of waste, and the model will classify it as either Organic or Recyclable.**
    """
)

# File uploader with drag-and-drop support
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False, help="Upload an image of waste for classification")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Classification button
    if st.button("Classify", key="classify_button"):
        with st.spinner("Classifying..."):
            prediction = predict_image(image_cv)

        # Display the classification result
        st.success(f"The image is classified as: **{prediction}**")

        # Trigger celebration animation
        rain(
            emoji="💥" if prediction == "Recyclable Waste" else "🌿",
            font_size=40,
            falling_speed=5,
            animation_length="infinite"
        )

        # Provide additional details
        if prediction == "Recyclable Waste":
            st.info(
                """
                **Recyclable Waste**: This type of waste includes materials like plastics, metals, paper, and glass that can be processed and reused.
                Proper recycling helps reduce landfill waste and conserves natural resources.
                """
            )
        elif prediction == "Organic Waste":
            st.info(
                """
                **Organic Waste**: This type of waste includes food scraps, yard waste, and other biodegradable materials.
                Organic waste can be composted to create nutrient-rich soil, which is beneficial for agriculture and gardening.
                """
            )

       # Store classification result in history
        if "history" not in st.session_state:
            st.session_state.history = []
        
        st.session_state.history.append({"filename": uploaded_file.name, "prediction": prediction})





# Sidebar History Section
st.sidebar.title("📜 Classification History")
if "history" in st.session_state and st.session_state.history:
    for idx, entry in enumerate(st.session_state.history[::-1]):
        st.sidebar.write(f"**{idx + 1}. {entry['filename']}**")
        st.sidebar.write(f"➡️ *{entry['prediction']}* ")
        if entry['prediction'] == "Recyclable Waste":
            st.sidebar.write("♻️ Materials: Plastics, Metals, Paper, Glass")
        elif entry['prediction'] == "Organic Waste":
            st.sidebar.write("🌿 Materials: Food Scraps, Yard Waste")
        st.sidebar.markdown("---")
else:
    st.sidebar.write("No classifications yet.")

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 14px;">
        Developed by <b>Naresh Thota</b> | Learn more about sustainability and be eco-friendly! 🌍 wait for out future solutions 
        <a href="https://example.com" target="_blank">our website</a>
    </div>
    """,
    unsafe_allow_html=True
)
