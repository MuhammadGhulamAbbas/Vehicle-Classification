"""
Vehicle Classifier - Streamlit deployment
=========================================
Loads the MobileNetV2 model trained in the Kaggle notebook and predicts the
vehicle class of any image you upload (or take with the camera).

Files expected next to this script (both are created by the notebook, section 10):
    vehicle_mobilenetv2_final.pt   - trained weights (state_dict)
    class_names.json               - list of class names, in training order

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

# --------------------------------------------------------------------------- config
APP_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = APP_DIR / "vehicle_mobilenetv2_final.pt"
DEFAULT_CLASSES = APP_DIR / "class_names.json"
IMAGE_SIZE = 224                                           # must match training
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet statistics (same as training)
DROPOUT = 0.3                                              # must match training (inactive at inference)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),           # same direct resize as the training pipeline
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# --------------------------------------------------------------------------- model
def build_model(n_classes: int) -> nn.Module:
    """Same architecture as the notebook: MobileNetV2 + new classifier head."""
    model = models.mobilenet_v2(weights=None)              # weights come from our checkpoint, no download
    model.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(model.classifier[1].in_features, n_classes))
    return model


@st.cache_resource(show_spinner="Loading model...")
def load_model(weights_bytes: bytes, n_classes: int) -> nn.Module:
    state = torch.load(io.BytesIO(weights_bytes), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = build_model(n_classes)
    model.load_state_dict(state, strict=True)              # fails loudly if classes/architecture differ
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model: nn.Module, image: Image.Image) -> torch.Tensor:
    """Return softmax probabilities (n_classes,) for one PIL image."""
    image = ImageOps.exif_transpose(image).convert("RGB")  # fix phone-photo rotation, drop alpha channel
    batch = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    return torch.softmax(model(batch), dim=1)[0].cpu()


# --------------------------------------------------------------------------- UI
st.set_page_config(page_title="Vehicle Classifier", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Classifier")
st.caption("MobileNetV2 transfer learning · trained with class balancing + Albumentations augmentation")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Show top-K classes", 1, 10, 5)
    threshold = st.slider("Minimum confidence", 0.0, 1.0, 0.40, 0.05,
                          help="Below this the app says 'uncertain' instead of guessing. "
                               "The model was trained with label smoothing, so confidences are naturally a bit lower.")
    st.divider()
    st.subheader("Model files")
    weights_up = st.file_uploader("Weights (.pt)", type=["pt", "pth"], help="Only needed if the file is not next to app.py")
    classes_up = st.file_uploader("class_names.json", type=["json"])

# ---- resolve weights and class names (uploaded files win over local files) ----
if weights_up is not None:
    weights_bytes = weights_up.getvalue()
elif DEFAULT_WEIGHTS.exists():
    weights_bytes = DEFAULT_WEIGHTS.read_bytes()
else:
    st.error(f"Weights not found. Put `{DEFAULT_WEIGHTS.name}` next to app.py or upload it in the sidebar.")
    st.stop()

if classes_up is not None:
    class_names = json.loads(classes_up.getvalue().decode("utf-8"))
elif DEFAULT_CLASSES.exists():
    class_names = json.loads(DEFAULT_CLASSES.read_text(encoding="utf-8"))
else:
    st.error(f"Class list not found. Put `{DEFAULT_CLASSES.name}` next to app.py or upload it in the sidebar.")
    st.stop()

try:
    model = load_model(weights_bytes, len(class_names))
except Exception as error:
    st.error("Could not load the model. The weights and class_names.json must come from the same training run.")
    st.exception(error)
    st.stop()

st.sidebar.success(f"Model ready · {len(class_names)} classes · {DEVICE.type.upper()}")

# ---- input: upload or camera ----
tab_upload, tab_camera = st.tabs(["📁 Upload images", "📷 Camera"])
with tab_upload:
    files = st.file_uploader("Choose one or more vehicle images", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True)
with tab_camera:
    shot = st.camera_input("Take a photo")
inputs = [(f.name, f) for f in files] + ([("camera.jpg", shot)] if shot is not None else [])

if not inputs:
    st.info("Upload an image (or use the camera) to get a prediction.")
    st.stop()

# ---- predict ----
summary_rows = []
for name, file in inputs:
    try:
        image = Image.open(file)
        probs = predict(model, image)
    except Exception as error:
        st.warning(f"Could not read `{name}`: {error}")
        continue

    k = min(top_k, len(class_names))
    conf, idx = probs.topk(k)
    best_class, best_conf = class_names[idx[0]], float(conf[0])
    summary_rows.append({"image": name, "prediction": best_class if best_conf >= threshold else "uncertain",
                         "confidence": round(best_conf, 4)})

    left, right = st.columns([1, 1.2])
    with left:
        st.image(ImageOps.exif_transpose(image), caption=name, use_container_width=True)
    with right:
        if best_conf >= threshold:
            st.success(f"### {best_class}\nConfidence: **{best_conf:.1%}**")
        else:
            st.warning(f"### Uncertain\nBest guess: **{best_class}** ({best_conf:.1%}), below the {threshold:.0%} threshold. "
                       "The image may show a vehicle type the model was not trained on.")
        chart = pd.DataFrame({"class": [class_names[i] for i in idx], "probability": [float(c) for c in conf]})
        st.bar_chart(chart.set_index("class"), horizontal=True)
    st.divider()

if len(summary_rows) > 1:
    st.subheader("Batch summary")
    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, use_container_width=True)
    st.download_button("Download predictions (CSV)", summary.to_csv(index=False), "predictions.csv", "text/csv")
