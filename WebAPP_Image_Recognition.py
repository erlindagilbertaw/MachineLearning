import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🍓 Cute Fruit AI",
    page_icon="🍑",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* MAIN BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: #e8f8f5; /* pastel green */
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #ffe4ec; /* pastel pink */
}

/* TITLE */
.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: #ff85a2;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #a88fac;
    font-size: 26px;
}

/* RESULT BOX */
.result-box {
    padding: 20px;
    border-radius: 20px;
    background: linear-gradient(135deg, #ffd6e0, #e0f7f4);
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #5a5a5a;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

/* BUTTON */
.stButton>button {
    background-color: #ffcad4;
    color: #5a5a5a;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #f4acb7;
}

/* SIDEBAR TITLE */
[data-testid="stSidebar"] h1 {
    font-size: 32px !important;
    color: #ff6f91;
}

/* SIDEBAR MENU */
[data-testid="stSidebar"] label {
    font-size: 22px !important;
}

/* GRID CARDS */
.grid-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}
.card {
    background: linear-gradient(135deg, #ffd6e0, #e0f7f4);
    padding: 12px;
    border-radius: 15px;
    text-align: center;
    font-size: 14px;
    font-weight: bold;
    color: #5a5a5a;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.keras")

# =========================
# PREDICTION FUNCTION
# =========================
def model_prediction(test_image):
    model = load_model()
    image = Image.open(test_image).resize((64, 64))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return predictions[0]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌸 MENU")
app_mode = st.sidebar.radio("", ["🏠 HOME", "📘 ABOUT", "🔮 PREDICT"])

# =========================
# HOME
# =========================
if app_mode == "🏠 HOME":
    st.markdown('<p class="title">🍓 Fruit & Veggie AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Cute AI that recognizes your fruits & veggies 🥕💖</p>', unsafe_allow_html=True)
    st.image("home_img.jpg", use_container_width=True)
    st.success("✨ Upload your image and let the AI guess it!")

# =========================
# ABOUT
# =========================
elif app_mode == "📘 ABOUT":
    st.markdown("## 📘 About This Project")
    st.markdown("### 📊 DATASET")

    # Fruits
    st.markdown("#### 🍓 Fruits")
    fruits_html = """
    <div class="grid-container">
        <div class="card">🍌 Banana</div>
        <div class="card">🍎 Apple</div>
        <div class="card">🍐 Pear</div>
        <div class="card">🍇 Grapes</div>
        <div class="card">🍊 Orange</div>
        <div class="card">🥝 Kiwi</div>
        <div class="card">🍉 Watermelon</div>
        <div class="card">🍈 Pomegranate</div>
        <div class="card">🍍 Pineapple</div>
        <div class="card">🥭 Mango</div>
    </div>
    """
    st.markdown(fruits_html, unsafe_allow_html=True)

    # Vegetables
    st.markdown("#### 🥕 Vegetables")
    vegetables_html = """
    <div class="grid-container">
        <div class="card">🥒 Cucumber</div>
        <div class="card">🥕 Carrot</div>
        <div class="card">🫑 Capsicum</div>
        <div class="card">🧅 Onion</div>
        <div class="card">🥔 Potato</div>
        <div class="card">🍋 Lemon</div>
        <div class="card">🍅 Tomato</div>
        <div class="card">🌱 Radish</div>
        <div class="card">🍠 Beetroot</div>
        <div class="card">🥬 Cabbage</div>
        <div class="card">🥗 Lettuce</div>
        <div class="card">🌿 Spinach</div>
        <div class="card">🌱 Soy Bean</div>
        <div class="card">🥦 Cauliflower</div>
        <div class="card">🫑 Bell Pepper</div>
        <div class="card">🌶️ Chilli Pepper</div>
        <div class="card">🥔 Turnip</div>
        <div class="card">🌽 Corn</div>
        <div class="card">🌽 Sweetcorn</div>
        <div class="card">🍠 Sweet Potato</div>
        <div class="card">🌶️ Paprika</div>
        <div class="card">🌶️ Jalapeño</div>
        <div class="card">🫚 Ginger</div>
        <div class="card">🧄 Garlic</div>
        <div class="card">🌱 Peas</div>
        <div class="card">🍆 Eggplant</div>
    </div>
    """
    st.markdown(vegetables_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Data Split
    st.markdown("### 📊 DATA SPLIT")
    c1, c2, c3 = st.columns(3)
    c1.metric("📚 Train", "100/images")
    c2.metric("🧪 Test", "10/images")
    c3.metric("🎯 Validation", "10/images")

# =========================
# PREDICTION
# =========================
elif app_mode == "🔮 PREDICT":
    st.markdown("## 🔍 Upload Your Image")
    test_image = st.file_uploader("📤 Choose an image", type=["jpg","png","jpeg"])

    # Inisialisasi session_state untuk tombol
    if "predicted_once" not in st.session_state:
        st.session_state.predicted_once = False

    if test_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(test_image, caption="Your Image 🧁", use_container_width=True)

        # Tombol Predict Now permanen disable setelah sekali klik
        predict_btn = st.button(
            "✨ Predict Now", 
            disabled=st.session_state.predicted_once
        )

        if predict_btn:
            # Set flag supaya tombol tidak bisa diklik lagi
            st.session_state.predicted_once = True

            with st.spinner("🌸 Thinking..."):
                probs = model_prediction(test_image)
                with open("labels.txt") as f:
                    labels = [line.strip() for line in f.readlines()]

                top_index = np.argmax(probs)
                prediction = labels[top_index]
                confidence = probs[top_index] * 100
                top3_idx = np.argsort(probs)[-3:][::-1]

            with col2:
                st.markdown(f"""
                <div class="result-box">
                PREDICTION <br><br>
                ✨ {prediction} ✨ <br><br>
                CONFIDENCE: {confidence:.2f}%
                </div>
                """, unsafe_allow_html=True)

                if confidence < 60:
                    st.warning("😢 Not very confident, try another image!")

                st.progress(int(confidence))

                st.markdown("### 🔮 Top 3 Results")
                for i in top3_idx:
                    st.write(f"💡 {labels[i]} : {probs[i]*100:.2f}%")
