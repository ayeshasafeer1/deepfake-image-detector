"""Deepfake image detector: Streamlit app.

Run with:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pipeline import center_crop, load_image, overlay, prepare

APP_DIR = Path(__file__).parent
SAMPLES_DIR = APP_DIR / "samples"
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_FILES = 12


def inject_styles() -> None:
    css = (APP_DIR / "assets" / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


st.set_page_config(
    page_title="Deepfake Image Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Loading model...")
def get_detector():
    from model_utils import WEIGHTS_PATH, Detector
    if not WEIGHTS_PATH.exists():
        return None
    return Detector()


detector = get_detector()
if detector is None:
    st.error("Model file not found.")
    st.markdown(
        "Copy **best.pt** and **final_settings.json** from your Google Drive folder "
        "`Deepfake_Project/final_model/` into this app's `model/` folder, then reload the page."
    )
    st.stop()


@st.cache_data(show_spinner="Analysing...", max_entries=64)
def analyze(raw: bytes, detect: bool, margin: float):
    """Run the full pipeline on one image. Cached, so changing display options is instant."""
    img = load_image(raw)
    model_img, info = prepare(img, detect=detect, margin=margin)
    p_real, cam = detector.predict(model_img, with_cam=True)
    thumb = img.copy()
    thumb.thumbnail((512, 512))
    return {
        "thumb": np.asarray(thumb),
        "model_img": np.asarray(model_img),
        "p_real": p_real,
        "cam": cam,
        "faces": info["faces"],
        "cropped": info["cropped"],
    }


def known_label(name: str):
    n = name.lower()
    return "REAL" if n.startswith("real") else "FAKE" if n.startswith("fake") else None


# ---------------------------------------------------------------- sidebar
trained_threshold = detector.threshold
with st.sidebar:
    st.markdown('<p class="dash-kicker">Analysis workspace</p>', unsafe_allow_html=True)
    st.header("Settings")
    detect = st.checkbox(
        "Detect and crop the face", value=True,
        help="The model was trained on 256x256 face crops. Turn this off only if your image is already a tight face crop.")
    margin = st.slider("Crop margin", 0.5, 1.2, 1.08, 0.05, disabled=not detect,
        help="How much of the surroundings to keep around the detected face. Larger = looser crop.")
    show_cam = st.checkbox("Show Grad-CAM heat map", value=True)
    opacity = st.slider("Heat map opacity", 0.1, 0.9, 0.45, 0.05, disabled=not show_cam)
    with st.expander("Advanced"):
        use_custom = st.checkbox("Use a custom decision threshold")
        custom = st.number_input(
            "Threshold on P(real)", min_value=0.001, max_value=0.999,
            value=float(trained_threshold), step=0.005, format="%.3f", disabled=not use_custom)
        st.caption(f"The trained threshold ({trained_threshold:.3f}) was chosen on held-out validation data. "
                   "Changing it here is for exploration only.")
    threshold = float(custom) if use_custom else trained_threshold
    st.divider()
    st.caption("Model: ResNet50 (ImageNet-pretrained), fine-tuned on face images.")
    st.caption(f"Decision rule: REAL if P(real) >= {threshold:.3f}, otherwise FAKE.")

# ------------------------------------------------------------------- main
with st.container(border=True):
    st.markdown('<p class="dash-kicker">Detection dashboard</p>', unsafe_allow_html=True)
    st.markdown(
    """
    <div class="app-title-row">
        <span class="face-icon">
            <svg xmlns="http://www.w3.org/2000/svg"
                 width="30" height="30"
                 viewBox="0 0 24 24"
                 fill="none"
                 stroke="currentColor"
                 stroke-width="1.8"
                 stroke-linecap="round"
                 stroke-linejoin="round">
                <path d="M7 3H5a2 2 0 0 0-2 2v2"/>
                <path d="M17 3h2a2 2 0 0 1 2 2v2"/>
                <path d="M21 17v2a2 2 0 0 1-2 2h-2"/>
                <path d="M3 17v2a2 2 0 0 0 2 2h2"/>
                <path d="M9 10h.01"/>
                <path d="M15 10h.01"/>
                <path d="M9 16c1.5 1.3 4.5 1.3 6 0"/>
            </svg>
        </span>
        <span>Deepfake Image Detector</span>
    </div>
    """,
    unsafe_allow_html=True
)

    st.caption(
        "Analyze images to estimate whether they are real or AI-generated."
    )
    st.info(
        "Disclaimer: AI detection is not 100% reliable. "
        "Results should be treated as an indication, not proof."
    )

tab_analyze, tab_how, tab_results = st.tabs(
    ["Analyze", "How it works", "Model & results"]
)

# ----------------------------------------------------------------- analyze

with tab_analyze:
    # Upload panel
    with st.container(border=True, key="upload_panel"):
        st.subheader("Upload images")
        st.caption(
            "Add images for AI-powered authenticity analysis. "
            "You can upload multiple images at once."
        )

        uploads = st.file_uploader(
            "Choose images or drag and drop them here",
            type=IMAGE_TYPES,
            accept_multiple_files=True,
            help="Supported formats: JPG, JPEG, PNG and WebP."
        )

        uploads = uploads or []

        st.caption(f"Upload limit: Up to {MAX_FILES} images per analysis.")

    # Sample images panel
    sample_names = sorted(
        p.name for p in SAMPLES_DIR.glob("*")
        if p.suffix.lower().lstrip(".") in IMAGE_TYPES
    )

    chosen_samples = []

    if sample_names:
        with st.container(border=True, key="sample_picker"):
            st.markdown(
                '<p class="upload-eyebrow">QUICK TEST</p>',
                unsafe_allow_html=True
            )
            st.subheader("Try sample images")
            st.caption(
                "Explore the detector using the images included with the project."
            )
            chosen_samples = st.multiselect(
                "Select sample images",
                sample_names,
                placeholder="Choose one or more sample images..."
            )

    # Keep the existing analysis input structure
    items = [(f.name, f.getvalue(), False) for f in uploads[:MAX_FILES]]
    items += [
        (n, (SAMPLES_DIR / n).read_bytes(), True)
        for n in chosen_samples
    ]

    if len(uploads) > MAX_FILES:
        st.warning(
            f"Only the first {MAX_FILES} uploaded files are analysed."
        )

    if not items:
        st.info(
            "Upload a photo to start. Best results: one clearly visible, "
            "front-facing face, no heavy filters, stickers or masks."
        )
    rows = []
    for name, raw, is_sample in items:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            try:
                res = analyze(raw, detect, float(margin))
            except Exception as exc:
                st.error(f"Could not analyse this file ({type(exc).__name__}). Is it a valid image?")
                continue

            p_real = res["p_real"]
            is_real = p_real >= threshold
            verdict = "REAL" if is_real else "FAKE or AI-modified"
            # Image and result layout
            if show_cam:
               col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1.2])
            else:
               col_a, col_b, col_d = st.columns([1, 1, 1.2])
            col_a.image(res["thumb"], caption="Uploaded photo")
            col_b.image(res["model_img"], caption="Model input (256x256)")
            if show_cam:
                heat = overlay(center_crop(Image.fromarray(res["model_img"])), res["cam"], opacity)
                col_c.image(heat, caption="Grad-CAM: evidence for FAKE (red = stronger)")
            with col_d:
                status_class = "real" if is_real else "fake"
                status_text = "REAL IMAGE" if is_real else "FAKE / AI-MODIFIED"
                st.markdown(
                     f"""
                     <div class="result-verdict {status_class}">
                         <span class="verdict-label">ANALYSIS RESULT</span>
                         <div class="verdict-title">{status_text}</div>
                     </div>
                     """,
                     unsafe_allow_html=True
                 )
                st.metric("P(real)", f"{p_real:.3f}", help="Raw model score, not a calibrated probability.")
                st.progress(float(min(max(p_real, 0.0), 1.0)), text=f"Model score | Threshold: {threshold:.3f}")
                if detect:
                    if res["faces"] == 0:
                        st.warning("No face found. The whole image was used so results may be less reliable.")
                    elif res["faces"] > 1:
                        st.info(f"{res['faces']} faces found; the largest one was analysed.")
                label = known_label(name) if is_sample else None
                if label:
                    agree = (label == "REAL") == is_real
                    st.caption(f"Known label (from file name): {label}. "
                               f"The model {'agrees' if agree else 'disagrees'}.")
            rows.append({"file": name, "verdict": verdict, "P(real)": round(p_real, 4),
                         "threshold": round(threshold, 4), "faces found": res["faces"]})

    if rows:
        with st.container(border=True):
            st.subheader("Summary")
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True)
            st.download_button("Download results (CSV)", df.to_csv(index=False), "results.csv", "text/csv")

# ------------------------------------------------------------------ how it works
with tab_how:
    with st.container(border=True):
        st.subheader("From photo to verdict")
        st.markdown(
        """
1. **Load** the image, fix its rotation and convert it to RGB.
2. **Find the face** (largest one) and crop around it with a margin, or skip this step if turned off.
3. **Resize to 256x256**, the size of the training images, then take the central **224x224** region.
4. **Normalise** the pixels and feed them to a **ResNet50** with a single output.
5. The output is turned into **P(real)**. The photo is called **REAL** if P(real) is at least the decision
   threshold, otherwise **FAKE or AI-modified**.
6. **Grad-CAM** shows which image regions pushed the score towards FAKE. It is coarse (7x7 cells) and
   shows where the model found evidence, not why.
        """)
    with st.container(border=True):
        st.subheader("How the model was trained")
        st.markdown(
        """
- **Data:** a subset of the Kaggle dataset *Deepfake and Real Images* (256x256 face images, real vs fake):
  40,000 training images plus 3,000 extra images from the same source as the final test set.
- **Model:** ImageNet-pretrained ResNet50, fine-tuned for 8 epochs with augmentation that varies blur,
  sharpness, JPEG compression and crops, so the model does not rely on image sharpness alone.
- **Selection:** checkpoint, blur strength and threshold were chosen on validation data, and the test set
  was evaluated once.
        """)

# ------------------------------------------------------------------ results
with tab_results:
    st.subheader("Held-out test results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy (threshold 0.5)", "94.8%")
    c2.metric("Test accuracy (tuned threshold)", "95.8%")
    c3.metric("Test AUC", "0.993")
    st.caption("Test set: 2,000 fake and 2,000 real images that were never used for training or tuning. "
               "It comes from the same source as 3,000 extra training images, so it measures performance "
               "on this kind of data, not on arbitrary photos.")

    with st.container(border=True):
        st.markdown("**Final model at the tuned threshold (0.079)**")
        st.dataframe(pd.DataFrame(
            {"Predicted fake": [1924, 93], "Predicted real": [76, 1907]},
            index=["Actually fake", "Actually real"]))

    with st.container(border=True):
        st.markdown("**All experiments**")
        results = pd.DataFrame(
        [
            ("Baseline: EffNet-B0, 10k images, basic aug", 95.5, None, 86.5, None, 0.952),
            ("A1: EffNet-B0, 40k, robust aug", 97.2, 84.1, 83.2, 88.5, 0.956),
            ("A2: A1 + test-time blur 1.0", 97.6, 91.4, 90.3, 92.9, 0.982),
            ("A3: ResNet50, 40k, robust aug", 96.1, 85.6, 85.7, 90.9, 0.974),
            ("C1: ResNet50 + extra test-source images", 96.6, 94.0, 93.5, 94.6, 0.990),
            ("C2: C1 + test-time blur 0.75", 97.2, 95.3, 94.7, 95.1, 0.991),
            ("Final: ResNet50, blurmix aug + extra images", 98.0, 95.5, 94.8, 95.8, 0.993),
        ],
        columns=["Run", "Validation acc (%)", "TestVal acc @0.5 (%)", "Test acc @0.5 (%)",
                 "Test acc tuned (%)", "Test AUC"])
        st.dataframe(results, hide_index=True)
        st.bar_chart(results.set_index("Run")[["Test acc @0.5 (%)", "Test acc tuned (%)"]])

st.divider()
st.caption("Semester project: deepfake image detection. ResNet50, PyTorch, Streamlit.")
