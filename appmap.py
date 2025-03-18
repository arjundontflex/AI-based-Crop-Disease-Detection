import os
import gdown
import requests
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAI
from streamlit_folium import folium_static
import folium
from geopy.geocoders import Nominatim

# Streamlit App Configuration
st.set_page_config(
    page_title="Crop Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State Initialization
if "detected_label" not in st.session_state:
    st.session_state["detected_label"] = None
if "detected_image" not in st.session_state:
    st.session_state["detected_image"] = None
if "show_diagnosis" not in st.session_state:
    st.session_state["show_diagnosis"] = False
if "conversation" not in st.session_state:
    st.session_state["conversation"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Google Drive Links for YOLO Models
rice_gdrive_link = "https://drive.google.com/uc?id=1_FDbnUGWtJRj4WuSuYeku1q-nhWHEHXz"
wheat_gdrive_link = "https://drive.google.com/uc?id=1nxktul7vEjszl9SRpmTKM0d4JoKtAXic"
maize_gdrive_link = "https://drive.google.com/uc?id=1pAt_0GucbhbuLdkqa1gVw8tHdBFaHH_T"

# Function to download models if missing
def download_model_if_missing(gdrive_link, model_path):
    if not os.path.exists(model_path):
        gdown.download(gdrive_link, model_path, quiet=False)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Failed to download {model_path}.")

# Load YOLO models
def load_models():
    rice_model_path = "rice_best.pt"
    wheat_model_path = "wheat_best.pt"
    maize_model_path = "maize_best.pt"

    download_model_if_missing(rice_gdrive_link, rice_model_path)
    download_model_if_missing(wheat_gdrive_link, wheat_model_path)
    download_model_if_missing(maize_gdrive_link, maize_model_path)

    return {
        "rice": YOLO(rice_model_path, task="detect"),
        "wheat": YOLO(wheat_model_path, task="detect"),
        "maize": YOLO(maize_model_path, task="detect"),
    }

# Load models globally
models = load_models()

# Initialize LLM and Memory
def initialize_llm():
    file_path = "Data.txt"
    if not os.path.exists(file_path):
        try:
            url = "https://drive.google.com/uc?id=1tx0Ax_-h1cPdnYteXTa4rDifx6ZYr3Bt"
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"Failed to download Data.txt: {e}")
            raise

    loader = TextLoader(file_path)
    documents = loader.load()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma.from_documents(documents, embeddings)

    llm = GoogleGenerativeAI(model="gemini-1.5-pro")
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    return conversation

st.session_state.conversation = initialize_llm()

# Function to detect disease
def detect_objects(image, model_choice):
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)
    try:
        model = models[model_choice]
        results = model.predict(temp_image_path, save=False, save_txt=False)
        annotated_image = results[0].plot()
        detected_labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        label_counts = Counter(detected_labels)
        most_common_label = (
            label_counts.most_common(1)[0][0] if label_counts else "No disease detected"
        )
        return Image.fromarray(annotated_image), most_common_label
    except Exception as e:
        st.error(f"Error during object detection: {e}")
        return None, "No disease detected"
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# Main UI
st.title("🌾 Crop Disease Detection")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
model_choice = st.sidebar.radio("Select Crop Type", options=["rice", "wheat", "maize"])

# Detect Disease Button
if uploaded_file:
    image = Image.open(uploaded_file)
    image.thumbnail((400, 400))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image")

    if st.sidebar.button("Detect Disease"):
        with st.spinner("Detecting... Please wait."):
            detected_image, detected_label = detect_objects(image, model_choice)
            st.session_state.detected_image = detected_image
            st.session_state.detected_label = detected_label
            st.session_state.show_diagnosis = False

    with col2:
        if st.session_state.detected_image:
            st.subheader("Detected Image")
            st.image(st.session_state.detected_image, caption="Detected Image")

    if st.session_state.detected_label and st.session_state.detected_label != "No disease detected":
        st.success(f"Detected Disease: {st.session_state.detected_label}")

        if st.button("Get Diagnosis"):
            st.session_state.show_diagnosis = True

        if st.session_state.show_diagnosis:
            query = f"Provide details about {st.session_state.detected_label}."
            with st.spinner("Fetching diagnosis..."):
                diagnosis = st.session_state.conversation.predict(input=query)
            st.subheader(f"Diagnosis for: {st.session_state.detected_label}")
            st.markdown(diagnosis)

# Nearby Plant & Pesticide Shops
st.title("🌱 Find Nearby Plant & Pesticide Shops")
location = st.text_input("Enter your location (e.g., Yelahanka, Bangalore)")

if location:
    geolocator = Nominatim(user_agent="crop_disease_detection_app")
    location_data = geolocator.geocode(location)

    if location_data:
        lat, lon = location_data.latitude, location_data.longitude
        st.success(f"Location found: {lat}, {lon}")

        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color="red")).add_to(m)

        folium_static(m)

        st.subheader("Nearby Shops:")
        st.write("- Example Plant Shop 1")
        st.write("- Example Pesticide Shop 2")
    else:
        st.error("Could not fetch location.")

# Chatbot Feature
st.sidebar.subheader("💬 Chatbot - Ask about crop diseases")
user_input = st.sidebar.text_input("Ask a question")

if user_input:
    response = st.session_state.conversation.run(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

for role, message in st.session_state.chat_history[-5:]:
    st.sidebar.write(f"**{role}:** {message}")
