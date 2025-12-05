import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.applications import efficientnet_v2, resnet_v2
from PIL import Image
import requests
import os
from huggingface_hub import hf_hub_download

HF_REPO = "david-arifin/OCT-Prediction"

MODEL_FILES = {
    "effnet": "effnet_model.keras",
    "resnet": "resnet_model.keras",
    "cnn": "cnn_model.keras",
    "weights": "ensemble_weights_3models.npy"
}

INPUT_SIZE = (224, 224)
NUM_CLASSES = 8
CLASS_NAMES = [
    "AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"
]

DEFAULT_TTA = 5
# MODEL_PATHS = {
#     "effnet": "effnet_model.keras",
#     "resnet": "resnet_model.keras",
#     "cnn": "cnn_model.keras", 
#     }
# WEIGHT_PATH = "ensemble_weights_3models.npy"

@st.cache_resource
def download_all_models():
    paths = {}
    for key, filename in MODEL_FILES.items():
        paths[key] = hf_hub_download(repo_id=HF_REPO, filename=filename)
    return paths

def load_all_models():
    # try:
    #     eff = keras.models.load_model(MODEL_PATHS["effnet"])
    # except Exception as e:
    #     st.error(f"Failed loading EfficientNet model: {e}")
    #     raise

    # try:
    #     res = keras.models.load_model(MODEL_PATHS["resnet"])
    # except Exception as e:
    #     st.error(f"Failed loading ResNet model: {e}")
    #     raise

    # try:
    #     cnn = keras.models.load_model(MODEL_PATHS["cnn"])
    # except Exception as e:
    #     st.error(f"Failed loading CNN model: {e}")
    #     raise

    # try:
    #     w = np.load(WEIGHT_PATH)
    #     w = w.astype(np.float32)
    #     if w.ndim == 1 and w.size == 3:
    #         w = w / np.sum(w)
    #     else:
    #         st.warning("Loaded PSO weights shape unexpected, normalizing anyway.")
    #         w = np.abs(w.flatten())[:3]
    #         w = w / np.sum(w)
    # except Exception as e:
    #     st.error(f"Failed loading PSO weights ({WEIGHT_PATH}): {e}")
    #     w = np.ones(3, dtype=np.float32) / 3.0

    paths = download_all_models()

    eff = keras.models.load_model(paths["effnet"])
    res = keras.models.load_model(paths["resnet"])
    cnn = keras.models.load_model(paths["cnn"])

    w = np.load(paths["weights"]).astype(np.float32)
    w = w / np.sum(w)

    return eff, res, cnn, w

effnet, resnet, cnn, PSO_W = load_all_models()

def make_tta_batch(base_img_norm, tta_steps=DEFAULT_TTA, seed=None):
    assert base_img_norm.ndim == 4 and base_img_norm.shape[0] == 1
    imgs = []
    rng = np.random.RandomState(seed) if seed is not None else None

    for _ in range(tta_steps):
        x = tf.convert_to_tensor(base_img_norm[0], dtype=tf.float32)

        if rng is None:
            do_flip = bool(np.random.randint(0, 2))
        else:
            do_flip = bool(rng.randint(0, 2))

        if do_flip:
            x = tf.image.flip_left_right(x)

        delta = 0.10
        if rng is None:
            rand_delta = np.random.uniform(-delta, delta)
        else:
            rand_delta = rng.uniform(-delta, delta)
        x = tf.image.adjust_brightness(x, rand_delta)
        x = tf.clip_by_value(x, 0.0, 1.0)

        imgs.append(x.numpy())

    batch = np.stack(imgs, axis=0).astype(np.float32)
    return batch

def predict_tta_batch(model, base_img_norm, preprocess_fn=None, tta_steps=DEFAULT_TTA, batch_size=None, seed=None):
    tta_batch = make_tta_batch(base_img_norm, tta_steps=tta_steps, seed=seed)  # (T,H,W,C) in [0,1]

    if preprocess_fn is not None:
        model_input = preprocess_fn(tta_batch * 255.0)
    else:
        model_input = tta_batch

    model_input = np.asarray(model_input, dtype=np.float32)

    preds = model.predict(model_input, batch_size=batch_size or len(model_input), verbose=0)
    mean_pred = preds.mean(axis=0)
    return mean_pred.astype(np.float32)

def ensemble_predict(base_img_norm, tta_steps=DEFAULT_TTA, seed=None):
    eff_pred = predict_tta_batch(effnet, base_img_norm, preprocess_fn=efficientnet_v2.preprocess_input, tta_steps=tta_steps, seed=seed)
    res_pred = predict_tta_batch(resnet, base_img_norm, preprocess_fn=resnet_v2.preprocess_input, tta_steps=tta_steps, seed=seed)
    cnn_pred = predict_tta_batch(cnn, base_img_norm, preprocess_fn=None, tta_steps=tta_steps, seed=seed)

    probs = PSO_W[0] * eff_pred + PSO_W[1] * res_pred + PSO_W[2] * cnn_pred
    probs = np.maximum(probs, 0.0)
    s = probs.sum()
    if s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return probs

def ai_generated_explanation_local(pred_class, probs):
    confidence = float(np.max(probs))
    confidence_text = (
        "high confidence" if confidence >= 0.80 else
        "moderate confidence" if confidence >= 0.60 else
        "low confidence"
    )

    base_explanations = {
        "AMD": "The model identifies disruptions in the retinal pigment epithelium and the presence of drusen-like deposits. These patterns commonly appear in Age-Related Macular Degeneration.",
        "CNV": "The prediction is driven by hyper-reflective areas and fluid-filled spaces that resemble choroidal neovascular membrane growth, which are characteristic signs of CNV.",
        "CSR": "Subretinal fluid pockets and dome-shaped neurosensory detachment are features often seen in Central Serous Retinopathy, which the model appears to have detected.",
        "DME": "The model is responding to cystoid spaces and increased retinal thickness, which frequently occur in Diabetic Macular Edema cases.",
        "DR": "The model likely focused on microaneurysm-like signals, uneven inner retinal layers, and reflectivity changes associated with Diabetic Retinopathy.",
        "DRUSEN": "Bright, nodular elevations beneath the retina are typical drusen formations, which are features the model appears to identify.",
        "MH": "A foveal disruption or gap-like pattern in the central retinal layers suggests the presence of a macular hole.",
        "NORMAL": "The retinal layers show smooth, continuous, and well-organized structure, which aligns with a healthy OCT scan."
    }

    base = base_explanations.get(pred_class, "The model detected structural patterns consistent with this condition.")

    generated = (
        f"**AI Explanation (local):**\n\n"
        f"The model predicts **{pred_class}** with **{confidence:.2f} confidence** "
        f"({confidence_text}). {base} "
        f"The decision is based on textural cues and layer morphology learned from the OCT dataset. "
        f"Although this analysis is automated, final interpretation should always be supported by clinical evaluation."
    )

    return generated

# StreamLit UI
st.set_page_config(page_title="OCT PSO Ensemble", layout="centered")
st.title("🔬 OCT Disease Classification — PSO Ensemble")

st.markdown("Upload an OCT image (jpg/jpeg/png). Models: EffNetV2 + ResNetV2 + custom CNN. PSO ensemble weights applied.")

col1, col2 = st.columns([2, 1])

with col2:
    tta_steps = DEFAULT_TTA
    show_topk = 3

uploaded_file = st.file_uploader("Upload OCT image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Cannot open image: {e}")
        raise

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.asarray(img_resized).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Running ensemble prediction..."):
        probs = ensemble_predict(arr, tta_steps=tta_steps, seed=None)

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]

    st.subheader("Prediction Result")
    st.markdown(f"**Predicted Class:** `{pred_class}` — confidence **{probs[pred_idx]:.4f}**")

    st.subheader(f"Top-{show_topk} predictions")
    topk_idx = np.argsort(probs)[::-1][:show_topk]
    for i, idx in enumerate(topk_idx, start=1):
        st.write(f"{i}. **{CLASS_NAMES[idx]}** — {probs[idx]:.4f}")

    st.subheader("All class probabilities")

    prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs})
    prob_df = prob_df.set_index("Class")
    st.bar_chart(prob_df)

    st.write(ai_generated_explanation_local(pred_class, probs))
else:
    st.info("Please upload an image to start inference.")

