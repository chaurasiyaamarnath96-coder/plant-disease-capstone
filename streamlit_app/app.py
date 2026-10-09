import streamlit as st
import sys
import tempfile

# -----------------------------
# IMPORT PROJECT MODULES
# -----------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import predict_image

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.main {
    background-color: #f4fff4;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #2E8B57;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

.footer {
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("🌱 Project Information")

    st.write("""
    **Model:** EfficientNet-B0

    **Classes:** 10 Tomato Diseases

    **Validation Accuracy:** 99.5%

    **Database:** SQLite

    **Framework:** Streamlit

    **Deployment:** Local Machine
    """)

    st.success("✅ Model Loaded Successfully")

# -----------------------------
# PAGE TITLE
# -----------------------------

st.markdown(
    """
    <div class="title">
        🌱 Tomato Disease Detection System
    </div>

    <div class="subtitle">
        AI Powered Plant Health Assistant
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# TWO COLUMN LAYOUT
# -----------------------------

left_col, right_col = st.columns([1,1])

# -----------------------------
# IMAGE UPLOAD
# -----------------------------

with left_col:

    st.subheader("📤 Upload Tomato Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

# -----------------------------
# IMAGE DISPLAY
# -----------------------------

if uploaded_file:

    with left_col:

        st.image(
            uploaded_file,
            caption="Uploaded Leaf Image",
            width=400
        )

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:

        temp_file.write(
            uploaded_file.read()
        )

        temp_file_path = temp_file.name

# -----------------------------
# PREDICTION BUTTON
# -----------------------------

    if st.button("🔍 Predict Disease"):

        disease, confidence, recommendation = predict_image(
            temp_file_path
        )

        clean_disease = disease.replace(
            "disease: ",
            ""
        ).strip()

        with right_col:

            st.markdown(
                """
                <div class="result-box">
                <h2 style='color:#2E8B57;'>
                Prediction Result
                </h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------
            # INVALID IMAGE
            # -------------------------

            if confidence < 0.98:

                st.error(
                    "⚠ Image may not be a valid tomato leaf."
                )

                st.write(
                    "Please upload a clear tomato leaf image."
                )

            # -------------------------
            # VALID IMAGE
            # -------------------------

            else:

                st.metric(
                    label="Detected Disease",
                    value=clean_disease
                )

                st.metric(
                    label="Confidence",
                    value=f"{confidence*100:.2f}%"
                )

                if confidence > 0.99:

                    st.success(
                        "✅ Very High Confidence"
                    )

                elif confidence > 0.95:

                    st.success(
                        "✅ High Confidence"
                    )

                else:

                    st.warning(
                        "⚠ Medium Confidence"
                    )

                st.markdown("---")

                st.subheader("💡 Recommendation")

                st.info(
                    recommendation
                )

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown(
    """
    <div class="footer">
    Developed using PyTorch • EfficientNet-B0 • Streamlit • SQLite
    </div>
    """,
    unsafe_allow_html=True
)