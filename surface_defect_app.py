
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io

# --------------------------------
# Load trained model
# --------------------------------
model = tf.keras.models.load_model(
    "/content/surface_defect_model.keras"
)

# --------------------------------
# Class names
# --------------------------------
class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

IMG_SIZE = (224, 224)

# --------------------------------
# Grad-CAM model
# --------------------------------
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer("out_relu").output,
        model.output
    ]
)

# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(
    page_title="Surface Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# --------------------------------
# Title
# --------------------------------
st.title("🔍 AI-Based Surface Defect Detection")

st.write(
    "An AI-powered system for detecting surface defects "
    "in steel images using deep learning."
)

st.divider()

# --------------------------------
# Upload image
# --------------------------------
st.subheader("📷 Upload Steel Surface Image")

uploaded_file = st.file_uploader(
    "Choose a steel surface image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# Prediction
# --------------------------------
if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    original_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    st.image(
        original_image,
        caption="Uploaded Steel Surface",
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🔍 Detect Defect",
        use_container_width=True
    ):

        with st.spinner("AI is analyzing the image..."):

            # ----------------------------
            # Prepare image
            # ----------------------------
            img = original_image.resize(
                IMG_SIZE
            )

            img_array = np.array(img)
            img_array = img_array.astype("float32") / 255.0
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # ----------------------------
            # Prediction + Grad-CAM
            # ----------------------------
            with tf.GradientTape() as tape:

                conv_output, predictions = grad_model(
                    img_array
                )

                predicted_index = tf.argmax(
                    predictions[0]
                )

                class_output = predictions[
                    :, predicted_index
                ]

            gradients = tape.gradient(
                class_output,
                conv_output
            )

            pooled_gradients = tf.reduce_mean(
                gradients,
                axis=(0, 1, 2)
            )

            conv_output = conv_output[0]

            heatmap = conv_output @ (
                pooled_gradients[..., tf.newaxis]
            )

            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(
                heatmap,
                0
            )

            heatmap = heatmap / (
                tf.reduce_max(heatmap)
                + tf.keras.backend.epsilon()
            )

            heatmap = heatmap.numpy()

            # ----------------------------
            # Prediction information
            # ----------------------------
            predicted_class = class_names[
                predicted_index
            ]

            confidence = (
                predictions[0][predicted_index]
                .numpy() * 100
            )

        # --------------------------------
        # Display result
        # --------------------------------
        st.success("✅ Analysis Completed!")

        st.subheader("🎯 Prediction Result")

        st.write(
            f"### Defect: "
            f"**{predicted_class.replace('_', ' ').title()}**"
        )

        st.write(
            f"### Confidence: "
            f"**{confidence:.2f}%**"
        )

        st.progress(
            float(
                predictions[0][predicted_index]
            )
        )

        st.divider()

        # --------------------------------
        # Prediction probabilities
        # --------------------------------
        st.subheader("📊 Prediction Probabilities")

        for i, class_name in enumerate(class_names):

            probability = (
                predictions[0][i].numpy() * 100
            )

            st.write(
                f"**{class_name.replace('_', ' ').title()}**: "
                f"{probability:.2f}%"
            )

            st.progress(
                float(predictions[0][i])
            )

        st.divider()

        # --------------------------------
        # Grad-CAM
        # --------------------------------
        st.subheader("🔥 AI Attention Map")

        st.write(
            "The heatmap highlights image regions "
            "that contributed to the model's prediction."
        )

        # Convert original image to OpenCV format
        original_np = np.array(
            original_image
        )

        # Resize heatmap
        heatmap_resized = cv2.resize(
            heatmap,
            (
                original_np.shape[1],
                original_np.shape[0]
            )
        )

        # Convert heatmap to color
        heatmap_uint8 = np.uint8(
            255 * heatmap_resized
        )

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        # Overlay
        overlay = cv2.addWeighted(
            original_np,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        st.image(
            overlay,
            caption="Grad-CAM: AI Attention Region",
            use_container_width=True
        )

else:

    st.info(
        "👆 Upload a steel surface image "
        "to begin defect detection."
    )
