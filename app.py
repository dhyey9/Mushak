import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor, AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor, AutoModelForCausalLM
from PIL import Image
import io
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import os

st.set_page_config(
    page_title="Mushak AI - Ganesh Chaturthi Companion", 
    page_icon="🐭",
    layout="wide"
)

# Load environment variables
load_dotenv()


genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="tunedModels/rolespecificconversationslordganesha-7kh",
    generation_config=generation_config
)

# Helper functions
def get_gemini_response(prompt):
    response = model.generate_content([prompt])
    return response.text.strip()
st.write("""
This AI-powered Mushak that is designed to enhance your celebration of Ganesha Chaturthi by providing you with detailed information about Lord Ganesha and related festival traditions. 
As an intelligent Assistant, I can provide you with these features:
- **Information Retrieval (Fine Tuned on Gemini Model)**: Get specific details about Lord Ganesha, including his attributes, stories, and various forms, upon request.
- **Festival Guide (Microsoft's MiniLM Model)**: Add information from the web and ask questions to enrich your knowledge base about Ganesha Chaturthi.
- **Text Analysis (Built on Facebook's Bart Large CNN Model)**: Summarize passages about Ganesha and get answers based on the provided text content.
- **Image Recognition (Built on Facebook's detr-resnet-50 Model)**: Upload images of Lord Ganesha to detect and identify objects, providing descriptions and relevant information.
- **Image Captioning (Built on Microsoft's git-base-coco)**: Generate appropriate captions for images related to Ganesha Chaturthi.
""")

# Initialize transformer models
@st.cache_resource
def load_object_detector():
    return pipeline("object-detection", model="facebook/detr-resnet-50")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")
# def load_summarizer():
#     config = AutoConfig.from_pretrained("facebook/bart-base")
#     config.max_position_embeddings = 2048  # Increase this value
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         "facebook/bart-base",
#         config=config,
#         ignore_mismatched_sizes=True  # Add this parameter
#     )
#     tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
#     return pipeline("summarization", model=model, tokenizer=tokenizer)

    

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/minilm-uncased-squad2")

@st.cache_resource
def load_image_captioner():
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    return model, processor
# def load_image_captioner():
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/git-base-coco")
#     processor = ViTImageProcessor.from_pretrained("microsoft/git-base-coco")
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/git-base-coco")
#     return model, processor, tokenizer




def detect_objects(image):
    object_detector = load_object_detector()
    results = object_detector(image)
    return results

def summarize_text(text):
    summarizer = load_summarizer()    
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def answer_question(context, question):
    qa_model = load_qa_model()
    answer = qa_model(question=question, context=context)
    return answer['answer']
def caption_image(image):
    model, processor = load_image_captioner()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption
# def caption_image(image):
#     image_captioner, image_processor, tokenizer = load_image_captioner()
#     inputs = image_processor(images=image, return_tensors="pt")
#     pixel_values = inputs.pixel_values
#     generated_ids = image_captioner.generate(pixel_values=pixel_values, max_length=50)
#     generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return generated_caption

# Streamlit UI
st.title("Mushak AI - Your Ganesh Chaturthi Companion")

# Sidebar for feature selection
st.sidebar.title("Features")

feature = st.sidebar.radio("Select a feature", 
    ["Chat with Mushak", "Story Summarization", "Question Answering", "Object Detection", "Image Captioning"])
    
if feature == "Chat with Mushak":
    st.header("Chat with Mushak about Ganesh Chaturthi")
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_input("Ask Mushak a question about Lord Ganesha:", key="user_input")
        if st.button("Generate", key="generate"):
            if user_input:
                response = get_gemini_response(user_input)
                st.session_state.chat_history.append((user_input, response))
                # st.write("Mushak's Response:", response)
                st.markdown(f"🐭 **Mushak's Response:** {response}")
            else:
                st.write("Please enter a prompt.")

        # Display selected chat history
        if 'selected_chat' in st.session_state:
            i = st.session_state.selected_chat
            st.write(f"Selected Chat: {st.session_state.chat_history[i][0]}")
            # st.write(f"Response: {st.session_state.chat_history[i][1]}")
            st.markdown(f"🐭 **Response:** {st.session_state.chat_history[i][1]}")

       
    with col2:
        st.subheader("Suggested Prompts")
        prompts = [
            "Good morning, can you give me some information on Ganesha?",
            "How is Ganesha related to other Hindu gods?",
            "What are the different names of Lord Ganesha?",
            "Why is Ganesha worshipped before starting any new venture?",
            "Why does Ganesha have one broken tusk?",
            "Why is Ganesha called the remover of obstacles?",
            "What are the symbols associated with Lord Ganesha?"
        ]
        
        selected_prompt = st.selectbox("Select a prompt", [""] + prompts, key="prompt_select")
        if selected_prompt:
            st.write(f"Selected prompt: {selected_prompt}")
            if st.button("Use this prompt"):
                response = get_gemini_response(selected_prompt)
                st.session_state.chat_history.append((selected_prompt, response))
                # st.write("Mushak's Response:", response)
                st.markdown(f"🐭 **Mushak's Response:** {response}")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
   

elif feature == "Story Summarization":
    st.header("Ganesha Story Summarization")
    story = st.text_area("Enter a story about Lord Ganesha:")
    if st.button("Summarize"):
        if story:
            summary = summarize_text(story)
            st.write("Summary:", summary)

elif feature == "Question Answering":
    st.header("Question Answering about Ganesh Chaturthi")
    context = st.text_area("Enter context about Ganesh Chaturthi:")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if context and question:
            answer = answer_question(context, question)
            st.write("Answer:", answer)

elif feature == "Object Detection":
    st.header("Object Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Objects"):
            detections = detect_objects(image)
            st.write(f"Number of objects detected: {len(detections)}")
            for detection in detections:
                st.write(f"Object: {detection['label']}, Confidence: {detection['score']:.2f}")

elif feature == "Image Captioning":
    st.header("Image Captioning")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Generate Caption"):
            caption = caption_image(image)
            st.write("Generated Caption:", caption)

st.sidebar.markdown("---")
st.sidebar.write("Mushak AI - Enhancing your Ganesh Chaturthi experience with AI")