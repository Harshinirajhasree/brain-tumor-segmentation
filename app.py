import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# Custom Loss Functions
# -----------------------------
def dice_loss(y_true, y_pred):
    smooth = 1.0

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) +
        tf.reduce_sum(y_pred_f) +
        smooth
    )

    return 1 - dice


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# -----------------------------
# Load Model
# -----------------------------
model = load_model(
    "brain_tumor_unet_v2_best.h5",
    custom_objects={
        "dice_loss": dice_loss,
        "bce_dice_loss": bce_dice_loss
    },
    compile=False
)

# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    layout="wide"
)

st.title("🧠 Brain Tumor Segmentation using U-Net")
st.write("Upload a brain MRI image to predict the tumor region.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["tif", "png", "jpg", "jpeg"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    resized = cv2.resize(img_rgb, (128, 128))
    resized = resized.astype(np.float32) / 255.0
    input_image = np.expand_dims(resized, axis=0)

    # Predict
    prediction = model.predict(input_image, verbose=0)[0]

    probability_map = prediction[:, :, 0]

    confidence = float(np.max(probability_map))

    mask = (probability_map > 0.5).astype(np.uint8) * 255

    tumor_pixels = int(np.count_nonzero(mask))

    # Resize mask to original image size
    mask_large = cv2.resize(
        mask,
        (img_rgb.shape[1], img_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Overlay
    overlay = img_rgb.copy()
    overlay[mask_large == 255] = [255, 0, 0]

    blended = cv2.addWeighted(
        img_rgb,
        0.7,
        overlay,
        0.3,
        0
    )

    # -----------------------------
    # Prediction Information
    # -----------------------------
    st.subheader("Prediction Information")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Confidence", f"{confidence:.3f}")

    with colB:
        st.metric("Tumor Pixels", tumor_pixels)

    with colC:
        if tumor_pixels > 0:
            st.success("Tumor Region Detected")
        else:
            st.info("No Tumor Detected")

    # -----------------------------
    # Display Images
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Original MRI")
        st.image(img_rgb)

    with col2:
        st.subheader("Probability Map")
        st.image(probability_map, clamp=True)

    with col3:
        st.subheader("Predicted Mask")
        st.image(mask_large)

    with col4:
        st.subheader("Tumor Overlay")
        st.image(blended)

    st.success("Segmentation Completed Successfully!")
