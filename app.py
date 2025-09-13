# app.py - Streamlit UI for Plant Species Identification
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import matplotlib.cm as cm

st.set_page_config(page_title="Plant Species Identification", layout="centered")

# ---------- DEFAULT PATHS / SETTINGS (edit if needed) ----------
DEFAULT_MODEL_PATH = "plant_species.h5"   # or "models/best_model.h5"
DEFAULT_LABELS_PATH = "labels.json"
DEFAULT_IMG_SIZE = 128

# ---------- CACHING helpers ----------
@st.cache_resource
def load_model_cached(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")

@st.cache_data
def load_labels_cached(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    # convert keys to ints if saved as strings
    return {int(k): v for k, v in raw.items()}

# ---------- Utility functions ----------
def preprocess_image_pil(pil_img: Image.Image, target_size: int, preprocess_option: str):
    img = pil_img.convert("RGB").resize((target_size, target_size))
    arr = np.asarray(img).astype("float32")
    if preprocess_option == "custom_0_1":
        arr = arr / 255.0
    elif preprocess_option == "EfficientNet":
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    elif preprocess_option == "ResNet50":
        arr = tf.keras.applications.resnet50.preprocess_input(arr)
    elif preprocess_option == "MobileNetV2":
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    else:
        arr = arr / 255.0
    return arr

def predict_topk(model, img_arr, k=3):
    x = np.expand_dims(img_arr, axis=0)
    preds = model.predict(x, verbose=0)[0]
    # if logits, convert to softmax
    if np.any(preds < 0) or preds.sum() <= 0:
        preds = tf.nn.softmax(preds).numpy()
    topk = preds.argsort()[-k:][::-1]
    return [(int(i), float(preds[i])) for i in topk]

def find_last_conv_layer(model):
    # preferred: explicit Conv2D
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found for Grad-CAM.")

def make_gradcam_heatmap(img_tensor, model, last_conv_name, pred_index=None):
    # img_tensor: 1xHxWx3 (preprocessed)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv = conv_outputs[0]
    heatmap = tf.matmul(conv, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha=0.4):
    # resize heatmap to image size and apply colormap
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(pil_img.size, resample=Image.BILINEAR)
    heatmap_arr = np.asarray(heatmap_img) / 255.0
    cmap = cm.get_cmap("jet")
    colored = cmap(heatmap_arr)[:,:,:3]
    colored_img = Image.fromarray(np.uint8(colored*255)).convert("RGBA")
    base = pil_img.convert("RGBA")
    blended = Image.blend(base, colored_img, alpha=alpha)
    return blended

# ---------- Sidebar controls ----------
st.sidebar.title("Settings")
model_path = st.sidebar.text_input("Model path (SavedModel dir or .h5)", value=DEFAULT_MODEL_PATH)
labels_path = st.sidebar.text_input("Labels JSON path (idx->label)", value=DEFAULT_LABELS_PATH)
img_size = st.sidebar.number_input("Image size (px)", value=DEFAULT_IMG_SIZE, step=1)
preproc = st.sidebar.selectbox("Preprocessing used in training", ["custom_0_1", "EfficientNet", "ResNet50", "MobileNetV2"])
top_k = st.sidebar.slider("Top K predictions", 1, 10, 3)
enable_gradcam = st.sidebar.checkbox("Enable Grad-CAM", value=True)
reload_btn = st.sidebar.button("Load / Reload model")

# ---------- Load model & labels ----------
model = None
labels = None
if model_path:
    try:
        model = load_model_cached(model_path)
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
if labels_path:
    try:
        labels = load_labels_cached(labels_path)
    except Exception as e:
        st.sidebar.error(f"Labels load error: {e}")

# ---------- Main page ----------
st.title("🌿 Plant Species Identification")
st.write("Upload a plant image (leaf/flower) and the model will predict the species.")

col1, col2 = st.columns([1,1])
with col1:
    uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    use_sample = st.checkbox("Use sample image (path)", value=False)
    if use_sample:
        sample_path = st.text_input("Local sample image path")
        if sample_path:
            try:
                with open(sample_path, "rb") as f:
                    uploaded_file = io.BytesIO(f.read())
            except Exception as e:
                st.error(f"Could not load sample image: {e}")

with col2:
    st.markdown("**Model info**")
    if model is not None:
        try:
            st.write(f"- Model loaded: `{model_path}`")
            st.write(f"- Input shape: {model.input_shape}")
        except Exception:
            st.write("- Model loaded")
    if labels is not None:
        st.write(f"- Classes loaded: {len(labels)}")

# ---------- Prediction flow ----------
if uploaded_file is not None:
    try:
        if isinstance(uploaded_file, io.BytesIO) or isinstance(uploaded_file, bytes):
            img = Image.open(io.BytesIO(uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file))
        else:
            img = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Failed to open image: {e}")
        img = None

    if img is not None:
        st.image(img, caption="Input image", use_column_width=True)
        if model is None or labels is None:
            st.error("Model or labels not loaded. Set paths in the sidebar.")
        else:
            # preprocess and predict
            img_arr = preprocess_image_pil(img, int(img_size), preproc)
            preds = predict_topk(model, img_arr, k=top_k)
            # map to labels
            results = []
            for idx, prob in preds:
                label = labels.get(idx, f"Class {idx}")
                results.append((label, prob))
            # show results
            st.markdown("### Predictions")
            st.success(f"Top: **{results[0][0]}** — {results[0][1]*100:.2f}%")
            table = [{"Rank": i+1, "Label": r[0], "Probability": f"{r[1]*100:.2f}%"} for i,r in enumerate(results)]
            st.table(table)

            # progress bars
            st.markdown("#### Confidence")
            for label, prob in results:
                st.write(f"{label} — {prob*100:.2f}%")
                st.progress(min(max(prob,0.0),1.0))

            # Grad-CAM (optional)
            if enable_gradcam and st.button("Generate Grad-CAM"):
                try:
                    last_conv = find_last_conv_layer(model)
                    x = np.expand_dims(img_arr, axis=0)
                    heatmap = make_gradcam_heatmap(x, model, last_conv, pred_index=preds[0][0])
                    overlay = overlay_heatmap(img, heatmap, alpha=0.45)
                    st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
else:
    st.info("Upload an image to get predictions.")

st.markdown("---")
st.caption("Ensure the app's preprocessing (selected in sidebar) matches the preprocessing used during model training.")
