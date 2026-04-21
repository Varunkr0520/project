import streamlit as st
import numpy as np
import joblib
from crop_guidelines import get_guidelines

# ----------------------------
# 🌄 BACKGROUND + UI STYLE (NEW)
# ----------------------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1500651230702-0e2d8a49d4ad");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Glass Cards */
.section-card {
    background: rgba(0,0,0,0.65);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    color: white;
}

/* Crop Cards */
.crop-card {
    background: rgba(0,0,0,0.75);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: white;
    font-size: 18px;
    text-align: center;
}

/* Headings */
h1, h2, h3 {
    color: white;
}

/* Inputs */
label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


def show_dashboard():
    st.title("🌾 Soil Analysis & Crop Recommendation System")

    # ----------------------------
    # SESSION STATE INIT
    # ----------------------------
    if "go_crop" not in st.session_state:
        st.session_state["go_crop"] = False

    if "predicted" not in st.session_state:
        st.session_state["predicted"] = False

    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    try:
        model = joblib.load("models/crop_model.pkl")
        encoders = joblib.load("models/encoders.pkl")
    except:
        st.error("❌ Model not found! Train model first.")
        return

    # ----------------------------
    # NAVIGATION
    # ----------------------------
    if st.session_state["go_crop"]:
        show_crop_page(
            st.session_state.get("fert"),
            st.session_state.get("soil"),
            st.session_state.get("ph")
        )
        return

    # ----------------------------
    # INPUT SECTION
    # ----------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("🌿 Enter Soil Parameters")

    N = st.number_input("Nitrogen (N)", 0.0)
    P = st.number_input("Phosphorus (P)", 0.0)
    K = st.number_input("Potassium (K)", 0.0)
    temp = st.number_input("Temperature", 0.0)
    humidity = st.number_input("Humidity", 0.0)
    ph = st.number_input("pH", 0.0)
    rainfall = st.number_input("Rainfall", 0.0)

    soil_type = st.selectbox("Soil Type", ["Clayey", "Sandy", "Loamy"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------
    # PREDICT
    # ----------------------------
    if st.button("Predict Soil Type"):
        st.session_state["predicted"] = True

        soil_col = None
        for key in encoders.keys():
            if "soil" in key.lower():
                soil_col = key
                break

        if soil_col is None:
            st.error("❌ Soil column not found in encoders")
            return

        soil_type_encoded = encoders[soil_col].transform([soil_type])[0]

        data = np.array([[N, P, K, temp, humidity, ph, rainfall, soil_type_encoded]])

        pred = model.predict(data)[0]
        output = encoders["target"].inverse_transform([pred])[0]

        st.session_state["output"] = output
        st.session_state["soil"] = soil_type
        st.session_state["ph"] = ph
        st.session_state["N"] = N
        st.session_state["P"] = P
        st.session_state["K"] = K

    # ----------------------------
    # RESULTS
    # ----------------------------
    if st.session_state.get("predicted"):

        output = st.session_state.get("output")
        soil_type = st.session_state.get("soil")
        ph = st.session_state.get("ph")
        N = st.session_state.get("N")
        P = st.session_state.get("P")
        K = st.session_state.get("K")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        if "-" in str(output):
            st.success(f"🧪 Recommended Fertilizer Ratio (N-P-K): {output}")

            n_val, p_val, k_val = output.split("-")

            st.info(f"""
• Nitrogen (N): {n_val}%  
• Phosphorus (P): {p_val}%  
• Potassium (K): {k_val}%  
""")

            if st.button("🌾 Show Suitable Crops"):
                st.session_state["go_crop"] = True
                st.session_state["fert"] = output
                st.rerun()

        else:
            st.success(f"🌱 Recommended Crop: {output}")

        st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------
        # SOIL ANALYSIS
        # ----------------------------
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("🌿 Soil Analysis")

        if N < 50:
            st.warning("Nitrogen is LOW → Add Urea / Compost")
        elif N > 100:
            st.warning("Nitrogen is HIGH")

        if P < 50:
            st.warning("Phosphorus is LOW → Add Bone Meal")
        elif P > 100:
            st.warning("Phosphorus is HIGH")

        if K < 50:
            st.warning("Potassium is LOW → Add Potash")
        elif K > 100:
            st.warning("Potassium is HIGH")

        if ph < 6:
            st.warning("Soil is ACIDIC")
        elif ph > 7.5:
            st.warning("Soil is ALKALINE")
        else:
            st.success("Soil pH is OPTIMAL ✅")

        st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------
        # GUIDELINES
        # ----------------------------
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("📘 Smart Guidelines")
        st.info(get_guidelines(output, soil_type, ph))

        st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------
        # GENERAL TIPS
        # ----------------------------
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("🌱 General Farming Tips")
        st.info("""
• Use organic compost  
• Maintain irrigation  
• Avoid overuse of fertilizers  
• Do soil testing  
""")

        st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------
# 🌾 CROP PAGE
# ----------------------------
def show_crop_page(fertilizer, soil_type, ph):

    st.title("🌾 Suitable Crops")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("📊 Soil Summary")
    st.write(f"Soil Type: {soil_type}")
    st.write(f"pH Level: {ph}")
    st.write(f"Fertilizer: {fertilizer}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🌱 Recommended Crops")

    crops = []

    if soil_type == "Clayey":
        crops = ["Rice", "Wheat"]
    elif soil_type == "Sandy":
        crops = ["Groundnut", "Millet", "Maize"]
    elif soil_type == "Loamy":
        crops = ["Sugarcane", "Cotton", "Vegetables"]

    if ph > 7.5:
        crops.append("Barley")
    elif ph < 6:
        crops.append("Potato")

    for crop in crops:
        st.markdown(f'<div class="crop-card">🌱 {crop}</div>', unsafe_allow_html=True)
        st.info(get_guidelines(crop, soil_type, ph))

    if st.button("⬅ Back to Dashboard"):
        st.session_state["go_crop"] = False
        st.rerun()