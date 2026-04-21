import streamlit as st
from database import create_db
from auth import register_user, login_user
from dashboard import show_dashboard

# ----------------------------
# INIT
# ----------------------------
create_db()

if "page" not in st.session_state:
    st.session_state.page = "home"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False




# ----------------------------
# 🌄 GLOBAL UI STYLE (FINAL FIXED)
# ----------------------------
st.markdown("""
<style>

/* 🌄 BACKGROUND */
.stApp {
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* 🌿 MAIN DASHBOARD CONTAINER (LIGHT GLASS) */
.main .block-container {
    background: rgba(0,0,0,0.25);   /* 🔥 LIGHT (image visible) */
    padding: 30px;
    border-radius: 15px;
    margin-top: 30px;
}

/* 🌿 HEADINGS */
h1, h2, h3 {
    background: rgba(0,0,0,0.35);   /* 🔥 LIGHT STRIP */
    padding: 10px 15px;
    border-radius: 10px;
    display: inline-block;
    color: white !important;
}

/* 🌿 LABEL TEXT */
label {
    color: white !important;
}

/* 🌿 INPUT FIELDS */
.stNumberInput, .stSelectbox {
    background: rgba(0,0,0,0.25);
    padding: 10px;
    border-radius: 10px;
}

/* 🌿 RESULT BOXES */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 12px;
    padding: 15px;
    font-size: 16px;
    background: rgba(0,0,0,0.35) !important;
    color: white !important;
}

/* 🌿 BUTTON STYLE */
.stButton>button {
    height: 55px;
    font-size: 18px;
    border-radius: 12px;
    border: none;
    color: white;
    background: linear-gradient(45deg, #2e7d32, #66bb6a);
    margin-top: 10px;
}

.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.3s;
    background: linear-gradient(45deg, #1b5e20, #81c784);
    box-shadow: 0px 6px 20px rgba(0,0,0,0.3);
}

/* 🌿 SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.5) !important;
}

/* 🌿 HOME / ABOUT BOX */
.box {
    background: rgba(0,0,0,0.45);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-top: 100px;
}

/* 🌿 ABOUT CARDS */
.card {
    background: rgba(0,0,0,0.45);
    padding: 25px;
    border-radius: 12px;
    color: white;
    margin: 20px auto;
    max-width: 800px;
}

/* 🌿 TITLE */
.title {
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    font-size: 18px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------
# 🌄 BACKGROUND FUNCTION (FINAL FIX)
# ----------------------------
def set_bg(image_url):
    st.markdown(f"""
    <style>

    .stApp {{
        background: 
            linear-gradient(rgba(0,0,0,0.20), rgba(0,0,0,0.20)),  /* 🔥 VERY LIGHT OVERLAY */
            url("{image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    </style>
    """, unsafe_allow_html=True)
# ----------------------------
# 🌾 HOME PAGE
# ----------------------------
def show_home():
    set_bg("https://images.unsplash.com/photo-1500382017468-9049fed747ef")

    st.markdown("""
    <div class="box">
        <div class="title">🌾 Smart Soil & Crop Recommendation System Using AI</div>
        <div class="subtitle">Empowering Farmers with Intelligent Decisions</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔐 Login"):
            st.session_state.page = "login"
            st.rerun()

    with col2:
        if st.button("📝 Register"):
            st.session_state.page = "register"
            st.rerun()

    with col3:
        if st.button("📘 About"):
            st.session_state.page = "about"
            st.rerun()


# ----------------------------
# 🔐 LOGIN PAGE
# ----------------------------
def show_login():
    set_bg("https://images.unsplash.com/photo-1464226184884-fa280b87c399")

    st.markdown("""
    <div class="box">
        <div class="title">🔐 Login</div>
    </div>
    """, unsafe_allow_html=True)

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(user, pwd):
            st.session_state.logged_in = True
            st.session_state.page = "dashboard"
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()


# ----------------------------
# 📝 REGISTER PAGE
# ----------------------------
def show_register():
    set_bg("https://images.unsplash.com/photo-1461354464878-ad92f492a5a0")
    st.markdown("""
    <div class="box">
        <div class="title">📝 Register</div>
    </div>
    """, unsafe_allow_html=True)

    user = st.text_input("Create Username")
    pwd = st.text_input("Create Password", type="password")

    if st.button("Register"):
        register_user(user, pwd)
        st.success("Registered Successfully ✅")

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()


# ----------------------------
# 📘 ABOUT PAGE
# ----------------------------
def show_about():
    set_bg("https://images.unsplash.com/photo-1471193945509-9ad0617afabf")

    st.markdown("""
    <div class="box">
        <div class="title">📘 About Project</div>
        <div class="subtitle">Smart Soil & Crop Recommendation System Using AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">🌾 <b>Project Description</b><br><br>This system helps farmers make intelligent decisions using AI by analyzing soil parameters and recommending suitable crops and fertilizers.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">🚀 <b>Features</b><br><br>• Crop Recommendation<br>• Fertilizer Suggestion (N-P-K)<br>• Soil Health Analysis<br>• Suitable Crop Guidance</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">🎯 <b>Objective</b><br><br>To improve agricultural productivity and help farmers choose the best crops based on soil conditions.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">🛠 <b>Technologies Used</b><br><br>Python • Streamlit • Machine Learning • Data Science</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">👩‍💻 <b>Developed By</b><br><br>Varun R'
    '</div>', unsafe_allow_html=True)

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()


# ----------------------------
# 📊 DASHBOARD (FIXED UI)
# ----------------------------
def show_main_dashboard():
    set_bg("https://images.unsplash.com/photo-1492496913980-501348b61469")
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    show_dashboard()

    st.markdown('</div>', unsafe_allow_html=True)

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "home"
        st.rerun()


# ----------------------------
# 🚀 ROUTING
# ----------------------------
if st.session_state.page == "home":
    show_home()

elif st.session_state.page == "login":
    show_login()

elif st.session_state.page == "register":
    show_register()

elif st.session_state.page == "about":
    show_about()

elif st.session_state.page == "dashboard":
    if st.session_state.logged_in:
        show_main_dashboard()
    else:
        st.session_state.page = "login"
        st.rerun()