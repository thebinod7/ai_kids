import numpy as np
import streamlit as st

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

from PIL import Image

def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image):
    if isinstance(image, str):
        img = Image.open(image) # If image is a file path load it
    else:
        img = image

    img = img.convert("RGB")  # Convert to RGB if not already
    img = img.resize((224, 224)) # Standard size for MobileNetV2
    arr = tf.keras.preprocessing.image.img_to_array(img) # Raw image array
    arr = np.expand_dims(arr, axis=0)    # Convert single photo into batch   
    return preprocess_input(arr)


def classify_image(model, image):
    try: 
        pp_image = preprocess_image(image)
        predictions = model.predict(pp_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0] # Get top 3 predictions
        return decoded_predictions
    
    except Exception as e:
        st.error(f"Error in classification: {e}")
        return None
    
def main():
    st.set_page_config(page_title="Image Classifier", page_icon=":guardsman:", layout="centered")
    st.title("Image Classifier")
    st.write("Upload an image to classify it and tell AI what is it.")
    
    @st.cache_resource
    def load_cached_model():
        return load_model()
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png",'webp'])

    if uploaded_file is not None:
        image = st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)
                print('predictions:', predictions) # [id, label, score]

                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}") # Two decimal places

if __name__ == "__main__":
    main()
               