# import streamlit as st
# import torch
# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, VisionEncoderDecoderModel, ViTImageProcessor
# from PIL import Image
# import io
# import google.generativeai as genai
# from dotenv import load_dotenv
# import os


# st.set_page_config(
#     page_title="Mushak AI - Ganesh Chaturthi Companion", 
#     page_icon="🐭",  # Mouse emoji as favicon
#     layout="wide"
# )
# # Load environment variables
# load_dotenv()

# # Configure the Gemini API
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# # Initialize transformer models
# @st.cache_resource
# def load_models():
#     object_detector = pipeline("object-detection", model="hustvl/yolos-tiny")
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", timeout=500)
#     qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
#     image_captioner = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning",timeout=500)
    
#     return object_detector, summarizer, translator, qa_model, image_captioner, image_processor, tokenizer

# object_detector, summarizer, translator, qa_model, image_captioner, image_processor, tokenizer = load_models()

# # Gemini model configuration
# generation_config = {
#     "temperature": 0.7,
#     "top_p": 1,
#     "max_output_tokens": 2048,
# }

# model = genai.GenerativeModel(
#     model_name="tunedModels/rolespecificconversationslordganesha-7kh",
#     generation_config=generation_config
# )

# # Helper functions
# def get_gemini_response(prompt):
#     response = model.generate_content([prompt])
#     return response.text.strip()

# def detect_objects(image):
#     results = object_detector(image)
#     return results

# def summarize_text(text):
#     summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
#     return summary[0]['summary_text']

# def translate_to_hindi(text):
#     translation = translator(text, max_length=200)
#     return translation[0]['translation_text']

# def answer_question(context, question):
#     answer = qa_model(question=question, context=context)
#     return answer['answer']

# def caption_image(image):
#     inputs = image_processor(images=image, return_tensors="pt")
#     pixel_values = inputs.pixel_values
#     generated_ids = image_captioner.generate(pixel_values=pixel_values, max_length=50)
#     generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return generated_caption

# # Streamlit UI
# st.title("Mushak AI - Your Ganesh Chaturthi Companion")

# # Sidebar for feature selection
# st.sidebar.title("Features")
# feature = st.sidebar.selectbox("Select a feature", 
#     ["Chat with Mushak", "Object Detection", "Story Summarization", 
#      "Hindi Translation", "Question Answering", "Image Captioning"])
    
# if feature == "Chat with Mushak":
#     st.header("Chat with Mushak about Ganesh Chaturthi")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         user_input = st.text_input("Ask Mushak a question about Lord Ganesha:", key="user_input")

#         if st.button("Generate", key="generate"):
#             if user_input:
#                 response = get_gemini_response(user_input)
#                 st.session_state.chat_history.append((user_input, response))
#                 # st.write("Mushak's Response:", response)
#                 st.markdown(f"🐭 **Mushak's Response:** {response}")
#             else:
#                 st.write("Please enter a prompt.")

#         # Display selected chat history
#         if 'selected_chat' in st.session_state:
#             i = st.session_state.selected_chat
#             st.write(f"Selected Chat: {st.session_state.chat_history[i][0]}")
#             # st.write(f"Response: {st.session_state.chat_history[i][1]}")
#             st.markdown(f"🐭 **Response:** {st.session_state.chat_history[i][1]}")

#     with col2:
#         st.subheader("Suggested Prompts")
#         prompts = [
#             "Good morning, can you give me some information on Ganesha?",
#             "How is Ganesha related to other Hindu gods?",
#             "What are the different names of Lord Ganesha?",
#             "Why is Ganesha worshipped before starting any new venture?",
#             "Why does Ganesha have one broken tusk?",
#             "Why is Ganesha called the remover of obstacles?",
#             "What are the symbols associated with Lord Ganesha?"
#         ]
        
#         selected_prompt = st.selectbox("Select a prompt", [""] + prompts, key="prompt_select")
#         if selected_prompt:
#             st.write(f"Selected prompt: {selected_prompt}")
#             if st.button("Use this prompt"):
#                 response = get_gemini_response(selected_prompt)
#                 st.session_state.chat_history.append((selected_prompt, response))
#                 # st.write("Mushak's Response:", response)
#                 st.markdown(f"🐭 **Mushak's Response:** {response}")
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []    


# elif feature == "Object Detection":
#     st.header("Object Detection")
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         if st.button("Detect Objects"):
#             detections = detect_objects(image)
#             st.write(f"Number of objects detected: {len(detections)}")
#             for detection in detections:
#                 st.write(f"Object: {detection['label']}, Confidence: {detection['score']:.2f}")

# elif feature == "Story Summarization":
#     st.header("Ganesha Story Summarization")
#     story = st.text_area("Enter a story about Lord Ganesha:")
#     if st.button("Summarize"):
#         if story:
#             summary = summarize_text(story)
#             st.write("Summary:", summary)

# elif feature == "Hindi Translation":
#     st.header("English to Hindi Translation")
#     text = st.text_area("Enter text to translate to Hindi:")
#     if st.button("Translate"):
#         if text:
#             hindi_text = translate_to_hindi(text)
#             st.write("Hindi Translation:", hindi_text)

# elif feature == "Question Answering":
#     st.header("Question Answering about Ganesh Chaturthi")
#     context = st.text_area("Enter context about Ganesh Chaturthi:")
#     question = st.text_input("Enter your question:")
#     if st.button("Get Answer"):
#         if context and question:
#             answer = answer_question(context, question)
#             st.write("Answer:", answer)

# elif feature == "Image Captioning":
#     st.header("Image Captioning")
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         if st.button("Generate Caption"):
#             caption = caption_image(image)
#             st.write("Generated Caption:", caption)

# st.sidebar.markdown("---")
# st.sidebar.write("Mushak AI - Enhancing your Ganesh Chaturthi experience with AI")

#-------------------
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

# Define the model
model = genai.GenerativeModel(
    model_name="tunedModels/rolespecificconversationslordganesha-7kh",
    generation_config=generation_config
)

# Function to generate a response
def get_gemini_response(prompt):
    response = model.generate_content([prompt])
    return response.text.strip()

# Function to save chat history
def save_chat_history(prompt, response):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append((prompt, response))

# Streamlit app interface
st.set_page_config(page_title="Mushak AI", page_icon="🐀", layout="wide")
st.title("Mushak AI - Your Guide to Lord Ganesha")

# Sidebar for chat history
with st.sidebar:
    st.header("Chat History")
    if 'chat_history' in st.session_state:
        for i, (p, r) in enumerate(st.session_state.chat_history):
            if st.button(f"Chat {i+1}: {p[:30]}...", key=f"history_{i}"):
                st.session_state.selected_chat = i

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input("Ask Mushak a question about Lord Ganesha:", key="user_input")

    if st.button("Generate", key="generate"):
        if user_input:
            response = get_gemini_response(user_input)
            save_chat_history(user_input, response)
            st.write("Mushak's Response:", response)
        else:
            st.write("Please enter a prompt.")

    # Display selected chat history
    if 'selected_chat' in st.session_state:
        i = st.session_state.selected_chat
        st.write(f"Selected Chat: {st.session_state.chat_history[i][0]}")
        st.write(f"Response: {st.session_state.chat_history[i][1]}")

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
            save_chat_history(selected_prompt, response)
            st.write("Mushak's Response:", response)
