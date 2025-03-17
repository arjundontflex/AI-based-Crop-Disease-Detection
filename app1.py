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
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

# Streamlit App Configuration
st.set_page_config(
    page_title="Crop Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to get content of Data.txt
def get_data_txt_content():
    file_path = "Data.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read()
    else:
        return "Data.txt content is unavailable."

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

# Google Drive Links for Models
rice_gdrive_link = "https://drive.google.com/uc?id=1_FDbnUGWtJRj4WuSuYeku1q-nhWHEHXz"
wheat_gdrive_link = "https://drive.google.com/uc?id=1nxktul7vEjszl9SRpmTKM0d4JoKtAXic"
maize_gdrive_link = "https://drive.google.com/uc?id=1pAt_0GucbhbuLdkqa1gVw8tHdBFaHH_T"

# Load YOLO Models
def download_model_if_missing(gdrive_link, model_path):
    """Download the model from Google Drive if it's not present locally."""
    if not os.path.exists(model_path):
        gdown.download(gdrive_link, model_path, quiet=False)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Failed to download {model_path}.")

def load_models():
    """Load YOLO models, downloading them if necessary."""
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

def detect_objects(image, model_choice):
    """Detect objects and return annotated image and most detected disease."""
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

# Single Page App
st.title("🌾 Crop Disease Detection")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Model Selection
model_choice = st.sidebar.radio("Select Crop Type", options=["rice", "wheat", "maize"])

# Detect Disease Button
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        max_size = (400, 400)
        image.thumbnail(max_size)

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

        # Show Detected Disease and Diagnosis Section
        if st.session_state.detected_label and st.session_state.detected_label != "No disease detected":
            st.success(f"Detected Disease: {st.session_state.detected_label}")

            # Get Diagnosis Button
            if st.button("Get Diagnosis"):
                st.session_state.show_diagnosis = True

            if st.session_state.show_diagnosis:
                query = f"Provide details about {st.session_state.detected_label}. Format: Detected Disease, Causes, Treatment, Precautions"
                with st.spinner("Fetching diagnosis..."):
                    diagnosis = st.session_state.conversation.predict(input=query)
                st.subheader(f"Diagnosis for: {st.session_state.detected_label}")
                st.markdown(diagnosis)

                # Suggested Follow-Up Options
                st.write("Would you like to explore more?")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button("Usable Pesticides"):
                        with st.spinner("Fetching list of pesticides..."):
                            pesticides_info = st.session_state.conversation.predict(
                                input=f"How do the pesticides for this {st.session_state.detected_label} disease vary based on environmental conditions and regional agricultural practices?"
                            )
                            st.subheader("List of Pesticides")
                            st.markdown(pesticides_info)

                with col2:
                    if st.button("Detailed Causes and Effects"):
                        with st.spinner("Fetching detailed causes and effects..."):
                            detailed_info = st.session_state.conversation.predict(
                                input=f"What are the long-term ecological and economic impacts of this {st.session_state.detected_label} disease on farming communities, and how can they be mitigated?"
                            )
                            st.subheader("Detailed Causes and Effects")
                            st.markdown(detailed_info)

                with col3:
                    if st.button("Prevention Methods"):
                        with st.spinner("Fetching prevention methods..."):
                            prevention_info = st.session_state.conversation.predict(
                                input=f"What innovative strategies, beyond traditional methods, can be adopted to prevent the recurrence of this {st.session_state.detected_label} disease in staple crops?"
                            )
                            st.subheader("Prevention Methods")
                            st.markdown(prevention_info)

                with col4:
                    if st.button("Treatment Options"):
                        with st.spinner("Fetching treatment options..."):
                            treatment_info = st.session_state.conversation.predict(
                                input=f"How can the integration of biological, chemical, and technological treatments improve the sustainability and effectiveness of managing this {st.session_state.detected_label} disease?"
                            )
                            st.subheader("Treatment Options")
                            st.markdown(treatment_info)

        else:
            st.warning("No disease detected.")
    except Exception as e:
        st.error(f"Error processing image or detecting: {e}")

# Chatbot for Follow-Up Questions
st.sidebar.subheader("Chatbot for Clarification")
user_input = st.sidebar.text_input("Ask a follow-up question:")
if user_input:
    is_relevant_query = "plant" in user_input.lower() or any(
        kw in user_input.lower() for kw in ["crop", "disease", "agriculture", "fertilizer", "pesticide"]
    )
    if is_relevant_query:
        with st.spinner("Fetching response..."):
            response = st.session_state.conversation.predict(input=user_input)
            st.session_state.chat_history.append((user_input, response))
    else:
        response = "Sorry, I can't respond to this query."
        st.session_state.chat_history.append((user_input, response))

# Display Chat History
if st.session_state.chat_history:
    st.sidebar.subheader("Chat History")
    for question, answer in st.session_state.chat_history:
        st.sidebar.markdown(f"*You:* {question}")
        st.sidebar.markdown(f"*Bot:* {answer}")