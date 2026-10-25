"""
SWYNEX Internship Task 4 — Final AI Application
AI Graphic Design Classifier — Final Application (Streamlit Web App)

Architecture, Model, and Pipeline Facts:
  - Base Model : MobileNetV2 (ImageNet weights frozen) + GlobalAveragePooling2D + Dropout(0.2) + Dense(5, softmax)
  - Input Size : 160 x 160 pixels, 3 color channels (RGB)
  - Preprocess : Raw 0-255 float32 array; internal MobileNetV2 normalizer scales to [-1, 1]
  - Classes    : ["cinematic", "minimalist", "typography", "sports", "vintage"]
  - Features   : Image validation, classification button, style score distribution,
                 AI Design Insight Assistant (visual rationale, actionable suggestions,
                 dual-condition uncertainty detection), About, Evaluation & Limitations,
                 and Privacy & Ethics tabs.
"""

from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st

import insights

# ----------------------------------------------------------------------
# Application Settings & Constants
# ----------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model" / "graphic_design_classifier.keras"
EXAMPLES_DIR = Path(__file__).parent / "examples"
IMAGE_SIZE = (160, 160)
CLASS_NAMES = insights.CLASS_NAMES

ALLOWED_TYPES = ["jpg", "jpeg", "png"]
MAX_FILE_MB = 10


# ----------------------------------------------------------------------
# Model Loading (Cached for performance)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained classifier model...")
def load_classifier():
    """Load the trained Keras model. Returns (model, error_message)."""
    if not MODEL_PATH.exists():
        return None, (
            f"Model file not found at: `{MODEL_PATH}`\n\n"
            "Please ensure `graphic_design_classifier.keras` is placed inside the `model` folder."
        )
    try:
        import keras

        model = keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as error:  # noqa: BLE001
        return None, f"The model file could not be loaded: {error}"

    output_units = model.output_shape[-1]
    if output_units != len(CLASS_NAMES):
        return None, (
            f"Model output mismatch: expected {len(CLASS_NAMES)} classes, but the model has {output_units}."
        )
    return model, None


# ----------------------------------------------------------------------
# Image Validation & Preprocessing
# ----------------------------------------------------------------------
def open_image(uploaded_file):
    """Validate and open the uploaded image file. Returns (image, error_message)."""
    if uploaded_file is None:
        return None, "No file was uploaded."

    size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return None, f"The file is {size_mb:.1f} MB. Please upload an image under {MAX_FILE_MB} MB."

    try:
        Image.open(uploaded_file).verify()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None, "This file is not a valid or readable JPG/PNG image."

    if image.width < 32 or image.height < 32:
        return None, f"The image is too small ({image.width} x {image.height} px). Minimum size is 32 x 32 px."

    return image, None


def preprocess(image: Image.Image) -> np.ndarray:
    """
    Match training pipeline:
      RGB conversion -> nearest-neighbor resize to (160, 160) -> float32 array in [0, 255]
    MobileNetV2 normalizer (x / 127.5 - 1) is inside the model, so we do not rescale by 255.
    """
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize(IMAGE_SIZE, Image.NEAREST)
    array = np.asarray(image_resized, dtype="float32")
    return np.expand_dims(array, axis=0)


def predict(model, image: Image.Image) -> np.ndarray:
    """Run model inference and validate output tensor."""
    batch = preprocess(image)
    raw_predictions = model.predict(batch, verbose=0)

    if raw_predictions is None or len(raw_predictions) == 0:
        raise ValueError("Model returned an empty prediction array.")

    probabilities = raw_predictions[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class scores, but model produced {len(probabilities)}."
        )
    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError("Model produced non-finite (NaN or Inf) output values.")

    return probabilities


# ----------------------------------------------------------------------
# Streamlit Page Setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Graphic Design Classifier — Final Application",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Sidebar Navigation & Context
# ----------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/maniram02112006/SWYNEX-AI-Graphic-Design-Classifier/main/graphic_design_classifier_dataset.zip"
        if False else "https://img.shields.io/badge/SWYNEX-Internship%20Task%204-blue?style=for-the-badge",
        use_container_width=True,
    )
    st.title("🎨 Navigation")
    st.markdown("**SWYNEX Technologies Internship**")
    st.caption("Task 4: Final AI Application")

    st.markdown("---")
    st.markdown("### 🏷️ Target Design Styles")
    for style in CLASS_NAMES:
        st.markdown(f"- **{style.title()}**")

    st.markdown("---")
    st.markdown("### ⚙️ System Specifications")
    st.markdown("- **Backbone:** MobileNetV2 (Frozen)")
    st.markdown("- **Input Shape:** 160 × 160 × 3 (RGB)")
    st.markdown("- **Runtime:** Local CPU (Python 3.12)")
    st.markdown("- **Inference Privacy:** 100% Offline")

# ----------------------------------------------------------------------
# Application Header
# ----------------------------------------------------------------------
st.title("🎨 AI Graphic Design Classifier — Final Application")
st.markdown(
    "Welcome to the **AI Graphic Design Classifier**, developed as the capstone project "
    "for **SWYNEX Technologies Internship Task 4**. This application classifies poster and banner designs "
    "into five core aesthetics, delivers transparent **AI Design Insights**, detects multi-style ambiguity, "
    "and offers practical composition recommendations."
)

# Check model availability immediately
model, load_error = load_classifier()
if load_error:
    st.error(f"❌ {load_error}")
    st.stop()

# ----------------------------------------------------------------------
# Main Application Tabs
# ----------------------------------------------------------------------
tab_classifier, tab_about, tab_evaluation, tab_ethics = st.tabs([
    "🖼️ Classifier & AI Assistant",
    "ℹ️ About the Model",
    "📊 Evaluation & Limitations",
    "🛡️ Privacy, Security & Ethics",
])

# ======================================================================
# TAB 1: CLASSIFIER & AI ASSISTANT
# ======================================================================
with tab_classifier:
    st.markdown("### 1. Select or Upload an Image")

    input_mode = st.radio(
        "Choose image input method:",
        ["Upload your own image", "Try an example from the dataset"],
        horizontal=True,
    )

    image = None
    image_caption = ""

    if input_mode == "Upload your own image":
        uploaded = st.file_uploader(
            "Upload a poster, artwork, or banner (JPG, JPEG, or PNG — Max 10 MB):",
            type=ALLOWED_TYPES,
        )
        if uploaded is not None:
            image, image_error = open_image(uploaded)
            if image_error:
                st.error(f"⚠️ {image_error}")
                image = None
            else:
                image_caption = uploaded.name
        else:
            st.info("👆 Upload an image file above to begin.")
    else:
        sample_files = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
        if not sample_files:
            st.warning("No sample images were found in the `examples/` directory.")
        else:
            sample_names = [f.name for f in sample_files]
            chosen = st.selectbox("Select a verified dataset sample:", sample_names)
            chosen_path = EXAMPLES_DIR / chosen
            try:
                image = Image.open(chosen_path)
                image.load()
                image_caption = f"Dataset Sample: {chosen}"
            except Exception as err:  # noqa: BLE001
                st.error(f"Could not load sample image: {err}")
                image = None

    if image is not None:
        st.markdown("---")
        col_preview, col_action = st.columns([1, 1], gap="large")

        with col_preview:
            st.subheader("🖼️ Image Preview")
            st.image(image, caption=image_caption, use_container_width=True)
            st.caption(f"Original Resolution: {image.width} × {image.height} px | Format: {image.format or 'RGB'}")

        with col_action:
            st.subheader("⚡ Classification Control")
            st.write(
                "Click the button below to preprocess the image, execute MobileNetV2 model inference, "
                "and generate rule-based design insights."
            )

            # Classify button state handling
            classify_btn = st.button("🔍 Classify Graphic Design Style", type="primary", use_container_width=True)

            if "classified_image" not in st.session_state:
                st.session_state.classified_image = None
            if "last_results" not in st.session_state:
                st.session_state.last_results = None

            # Reset state if user changes the image
            if st.session_state.classified_image != image_caption:
                st.session_state.classified_image = None
                st.session_state.last_results = None

            if classify_btn:
                with st.spinner("Classifying design style and computing insights..."):
                    try:
                        probabilities = predict(model, image)
                        analysis = insights.analyze_design(probabilities, CLASS_NAMES)
                        st.session_state.classified_image = image_caption
                        st.session_state.last_results = (probabilities, analysis)
                    except Exception as error:  # noqa: BLE001
                        st.error(f"❌ Prediction failed: {error}")

            if st.session_state.last_results is not None:
                probabilities, analysis = st.session_state.last_results

                predicted_style = analysis["predicted_class"].title()
                confidence_pct = analysis["confidence"] * 100

                st.markdown(f"### Predicted Style: **{predicted_style}**")
                st.metric(
                    label="Model Confidence Score",
                    value=f"{confidence_pct:.1f}%",
                    help="Softmax output score across the 5 closed style categories.",
                )

                # Uncertainty / Ambiguity Warning Box
                if analysis["is_uncertain"]:
                    with st.container():
                        st.warning("⚠️ **Uncertainty / Ambiguity Detected**")
                        for reason in analysis["uncertainty_reasons"]:
                            st.markdown(f"- {reason}")
                        st.info(
                            "💡 **Recommendation:** Review this design manually. It may represent a hybrid blend "
                            "of multiple styles or diverge from the training sample distribution."
                        )

                # Class Probability Breakdown
                st.markdown("#### Probability Breakdown Across All 5 Styles")
                for name, prob in analysis["ranked_classes"]:
                    st.write(f"**{name.title()}**: {prob * 100:.1f}%")
                    st.progress(float(min(max(prob, 0.0), 1.0)))

        # --------------------------------------------------------------
        # Section: AI Design Insight Assistant
        # --------------------------------------------------------------
        if st.session_state.last_results is not None:
            _, analysis = st.session_state.last_results
            st.markdown("---")
            st.subheader("💡 AI Design Insights Assistant")
            st.markdown(
                "The **AI Design Insight Assistant** uses transparent, deterministic heuristics "
                "to explain why the visual composition matches the predicted category and delivers "
                "practical creative recommendations."
            )

            col_insight1, col_insight2 = st.columns([1, 1], gap="large")

            with col_insight1:
                st.markdown(f"#### 🔍 Visual Rationale: {analysis['style_title']}")
                st.info(f"**Core Profile:** {analysis['style_summary']}")
                st.write(analysis["visual_rationale"])

                if analysis["hybrid_blend_note"]:
                    st.warning(analysis["hybrid_blend_note"])

            with col_insight2:
                st.markdown("#### 🛠️ Design Improvement Suggestions")
                st.write("Actionable design tips tailored to elevate this specific visual aesthetic:")
                for idx, suggestion in enumerate(analysis["suggestions"], start=1):
                    st.markdown(f"**{idx}.** {suggestion}")

            st.caption(f"📌 **Interpretation Notice:** {analysis['disclaimer']}")

# ======================================================================
# TAB 2: ABOUT THE MODEL & ARCHITECTURE
# ======================================================================
with tab_about:
    st.markdown("### ℹ️ About the Model & Architecture")
    st.markdown(
        """
This application integrates the deep learning model originally trained in **Task 1** 
and operationalized in **Task 2** and **Task 3**.

#### 1. Model Architecture
- **Base Network:** `MobileNetV2` with ImageNet pre-trained feature extraction weights (frozen during training).
- **Pooling Layer:** `GlobalAveragePooling2D` (reduces 2D spatial feature maps into a 1D vector).
- **Regularization:** `Dropout(0.2)` to reduce overfitting.
- **Classification Head:** `Dense(5, activation='softmax')` producing probabilities across the 5 styles.
- **Artifact Size:** 9,698,107 bytes (~9.7 MB in Keras 3 format).

#### 2. Input Specifications & Preprocessing Pipeline
- **Input Dimensions:** `160 × 160 × 3` (RGB color channels).
- **Resize Algorithm:** Nearest-neighbor interpolation.
- **Pixel Range:** Raw `0–255` float32 array.
- **Normalization Fact:** The MobileNetV2 preprocessing layer (`x / 127.5 - 1`) is embedded **inside** the saved `.keras` model. 
  The application strictly passes raw `[0, 255]` values and does **not** divide by 255, preserving the exact numerical distribution used during training.

#### 3. Complete End-to-End AI Workflow
```
[User Image Upload]
        │
        ▼
[Input Validation] ──(Size check <10MB, PIL verify, min 32x32 px)
        │
        ▼
[Preprocessing] ────(Convert RGB, resize 160x160, float32 tensor)
        │
        ▼
[Model Inference] ──(MobileNetV2 forward pass -> 5 softmax scores)
        │
        ▼
[Score Analysis] ───(Validate finite values, rank probabilities)
        │
        ▼
[Uncertainty Check]─(Score < 50% OR Margin between #1 and #2 < 15%)
        │
        ▼
[AI Insights Engine](Generate visual rationale & 3 actionable tips)
        │
        ▼
[Streamlit UI Display]
```
"""
    )

# ======================================================================
# TAB 3: EVALUATION & DATASET LIMITATIONS
# ======================================================================
with tab_evaluation:
    st.markdown("### 📊 Model Evaluation & Dataset Limitations")
    st.markdown(
        """
#### 1. Dataset Origin & Training Constraints
- The underlying classifier was trained in Task 1 on a **small educational sample dataset of 20 AI-generated images** (4 images per category).
- Because no independent, held-out test split was reserved during Task 1, **no generalized accuracy metric (e.g. 95% test accuracy) is claimed**.
- Predictions on the included sample images demonstrate pipeline execution and internal model consistency, but do **not** prove commercial generalization.

#### 2. Live Verified Evaluation Results
All five dataset sample images were evaluated using the automated evaluation script (`evaluation/evaluate_model.py`):

| Sample Image | Predicted Style | Top Confidence | Runner-Up Style | Margin | Status |
|---|---|---|---|---|---|
| `cinematic_01.jpg` | **Cinematic** | 87.5% | Sports (6.8%) | 80.6% | ✅ Confident |
| `minimalist_02.jpg` | **Minimalist** | 31.9% | Typography (22.0%) | 9.9% | ⚠️ Uncertain |
| `sports_01.jpg` | **Sports** | 91.3% | Cinematic (5.1%) | 86.2% | ✅ Confident |
| `typography_01.jpg` | **Typography** | 95.7% | Sports (1.8%) | 93.9% | ✅ Confident |
| `vintage_01.jpg` | **Vintage** | 97.8% | Sports (0.8%) | 97.0% | ✅ Confident |

#### 3. Uncertainty & Ambiguity Heuristics
- **Low Confidence Alert:** Triggered whenever the highest predicted score is below 50%.
- **Ambiguity Alert:** Triggered whenever the gap between the top prediction and runner-up is below 15%.
- Notice that `minimalist_02.jpg` properly triggers **both** uncertainty conditions (Confidence 31.9%, Margin 9.9%), alerting the user to treat the prediction with caution.

#### 4. Requirements for a Production Evaluation Benchmark
To transition this prototype into a production-grade classifier, future iterations require:
1. **Curated Benchmark Dataset:** 500+ diverse, human-designed posters and graphic layouts across all 5 styles.
2. **Stratified Split:** Strict 70% train, 15% validation, and 15% independent held-out test set.
3. **Formal Metrics:** Precision, Recall, Macro-F1 score, Confusion Matrix, and Expected Calibration Error (ECE).
4. **Out-of-Distribution (OOD) Rejection:** Ability to detect and reject non-poster inputs (e.g. selfies, landscapes).
"""
    )

# ======================================================================
# TAB 4: PRIVACY, SECURITY & ETHICS
# ======================================================================
with tab_ethics:
    st.markdown("### 🛡️ Privacy, Security & Responsible AI Considerations")
    st.markdown(
        """
#### 1. Privacy & Data Handling
- **100% Local Execution:** All image preprocessing, model inference, and heuristic insight generation occur entirely on your local machine.
- **No Cloud Transmissions:** No uploaded images, filenames, or user metadata are transmitted to external servers, cloud storage, or third-party APIs.
- **No Data Retention:** Uploaded images reside strictly in temporary RAM during the active session. When the tab or server closes, images are automatically freed.
- **No Secrets or Credentials:** This application requires zero API keys, passwords, or authentication tokens.

#### 2. Responsible AI Principles
- **Uncalibrated Softmax Scores:** Softmax output probabilities reflect internal mathematical optimization across 5 closed categories; they are **not** calibrated real-world probabilities.
- **Non-Authoritative Guidance:** The classifier output should be treated as directional, creative inspiration rather than an authoritative design verdict.
- **No Intellectual Property Claims:** The model classifies surface visual aesthetics only. It makes no claims regarding copyright status, original artist attribution, or legal ownership.
- **Dataset Bias Awareness:** Because the model was trained on illustrative AI-generated samples, it may exhibit bias toward specific color palettes or fail on unconventional artwork.
- **Human-in-the-Loop:** Creative judgment remains with the human designer. When uncertainty warnings are displayed, manual review is strongly advised.
"""
    )

# ----------------------------------------------------------------------
# Application Footer
# ----------------------------------------------------------------------
st.divider()
st.caption(
    "SWYNEX Technologies Internship — Task 4: Final AI Application | "
    "Developed by Maniram | Model: MobileNetV2 Keras 3 | Runtime: Python 3.12 (Local CPU)"
)
