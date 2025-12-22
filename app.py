import os
import streamlit as st
import torch
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from model_loader import load_model
from utils import load_classes, get_preprocess, predict_topk
from gradcam import GradCAM, overlay_cam_on_image


st.set_page_config(page_title="Raw Materials Classifier (ResNet101)", layout="wide")

# ✅ Hardcoded Project Root
PROJECT_ROOT = r"C:\Users\pro eiko\Desktop\Project AI"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ResNet101_finetuned_FINAL_BEST.pth")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "models", "classes.json")

# ✅ Correct default test folder (لاحظ /test في الآخر)
DEFAULT_TEST_DIR = os.path.join(PROJECT_ROOT, "test-20251222T063035Z-3-001", "test")


@st.cache_resource
def init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(CLASSES_PATH)
    model = load_model(MODEL_PATH, num_classes=len(classes), device=device)
    cam = GradCAM(model, target_layer=model.layer4)  # last conv block
    return device, classes, model, cam


device, classes, model, cam = init()


def get_eval_transform():
    # نفس Normalize القياسي لـ ResNet (ImageNet)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@st.cache_data
def evaluate_on_folder(test_dir: str, batch_size: int = 32):
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    dataset = datasets.ImageFolder(test_dir, transform=get_eval_transform())

    # ImageFolder بيرتب أبجديًا، فنعمل mapping لو ترتيب classes.json مختلف
    dataset_classes = dataset.classes
    if dataset_classes != classes:
        name_to_your_idx = {name: classes.index(name) for name in classes if name in dataset_classes}
        if len(name_to_your_idx) != len(dataset_classes):
            missing = set(dataset_classes) - set(classes)
            raise ValueError(f"Some folder classes not found in classes.json: {missing}")

        remap = np.zeros(len(dataset_classes), dtype=int)
        for i, name in enumerate(dataset_classes):
            remap[i] = name_to_your_idx[name]
    else:
        remap = None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.numpy()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            if remap is not None:
                labels = remap[labels]

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    summary = {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
        "num_samples": int(len(y_true)),
    }

    return report, cm, summary


st.title("Raw Materials Classification — ResNet101 (Fine-tuned)")
st.caption("Upload an image to get Top-3 predictions + Grad-CAM explainability.")

# =========================
# Inference + Grad-CAM
# =========================
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

img = Image.open(uploaded).convert("RGB")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    st.image(img, use_container_width=True)

    results, probs = predict_topk(model, img, classes, device, k=3)
    st.subheader("Top-3 Predictions")
    for name, p in results:
        st.write(f"**{name}** — {p*100:.2f}%")

with col2:
    st.subheader("Grad-CAM")
    x = get_preprocess()(img).unsqueeze(0).to(device)
    x.requires_grad_(True)

    top1_name, _ = results[0]
    top1_idx = classes.index(top1_name)

    cam_map = cam.generate(x, class_idx=top1_idx)
    overlay = overlay_cam_on_image(img, cam_map)

    st.image(overlay, caption=f"Attention Map for: {top1_name}", use_container_width=True)

st.divider()
st.write(f"Running on **{device}**")

# =========================
# Evaluation (Precision / Recall / Confusion Matrix)
# =========================
st.header("Model Evaluation (Test Set)")

with st.sidebar:
    st.subheader("Evaluation Settings")
    test_dir = st.text_input("Test folder path", value=DEFAULT_TEST_DIR)
    batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=32, step=1)

# ✅ Debug للتأكد إن الباث صح والفولدرات موجودة
st.subheader("Debug (Test folder)")
st.write("Test dir:", test_dir)
st.write("Exists:", os.path.isdir(test_dir))
if os.path.isdir(test_dir):
    st.write("Class folders found:",
             [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

# ✅ Checkbox بدل زرار عشان يفضل شغال ويعرض النتائج
run_eval = st.checkbox("Run Evaluation on Test Set", value=True)

if run_eval:
    try:
        report, cm, summary = evaluate_on_folder(test_dir, batch_size=int(batch_size))

        st.subheader("Summary Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{summary['accuracy']*100:.2f}%")
        c2.metric("Precision (Macro)", f"{summary['precision_macro']*100:.2f}%")
        c3.metric("Recall (Macro)", f"{summary['recall_macro']*100:.2f}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("F1 (Macro)", f"{summary['f1_macro']*100:.2f}%")
        c5.metric("Precision (Weighted)", f"{summary['precision_weighted']*100:.2f}%")
        c6.metric("Samples", f"{summary['num_samples']}")

        st.subheader("Classification Report")
        st.code(report, language="text")

        st.subheader("Confusion Matrix")
        fig = plt.figure(figsize=(7, 6))
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(classes)), labels=classes)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Evaluation failed: {e}")