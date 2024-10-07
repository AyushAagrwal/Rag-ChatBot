import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import google.generativeai as genai
import streamlit as st
from PIL import Image  # Import for handling images
import logging  # For logging errors and status
import tempfile  # To handle temporary file creation


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from dotenv import load_dotenv
import os

# Load variables from the .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv('API_KEY')

# Now you can use the API key in your GenAI configuration
genai.configure(api_key=api_key)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

class RAGSystem:
    def __init__(self, model):
        self.model = model
        self.memory = []  # For storing chat history
        self.chat_session = self.model.start_chat()  # Initialize chat session once

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        is_scanned_pdf = True  # We start by assuming it's a scanned PDF
        
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():  # If there's any text, treat it as a digital PDF
                    is_scanned_pdf = False  # It's a digital PDF if text is found
                    text += page_text
                else:
                    # No text found on the page; continue assuming it's scanned
                    pass
        except Exception as e:
            logging.error(f"Error processing PDF with PyMuPDF: {e}")

        # Fallback to OCR if PyMuPDF found no text or encountered an error
        if is_scanned_pdf or not text.strip():
            logging.info("No text found using PyMuPDF, applying OCR...")
            try:
                images = convert_from_path(pdf_path, poppler_path=r'D:\Placements\Assignments\Rag-ChatBOT\poppler\Library\bin')  # Convert PDF pages to images
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                # Apply OCR with specified languages: Urdu (ur), English (eng), Bengali (ben), Chinese (chi_sim)
                for image in images:
                    ocr_text = pytesseract.image_to_string(image, lang="eng+ben+chi_sim+urd")
                    text += ocr_text
            except Exception as e:
                logging.error(f"Error converting PDF to images or applying OCR: {e}")
        
        return text

    def query_decomposition(self, query):
        # Simple decomposition for this example
        return query.split("?")

    def optimized_chunking(self, text):
        # Example chunking: splitting by double new lines
        return text.split("\n\n")

    def hybrid_search(self, query):
        # Placeholder for hybrid search logic
        return [query]  # Just return the query for now

    def answer_query(self, query, context):
        # Construct a prompt using both the query and the extracted context
        prompt = f"The following is a text extracted from a PDF:\n\n{context}\n\nUser question: {query}\n\nProvide a relevant answer:"
        
        try:
            # Send the prompt to the model
            response = self.chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error in model response: {e}")
            return "Error generating response. Please try again."
        
    def add_to_memory(self, user_query, bot_response):
        self.memory.append({"user": user_query, "bot": bot_response})

    def get_memory(self):
        return self.memory


def main():
    st.title("📄 Multilingual PDF RAG ChatBot🤖")

    st.write(
        """
        This tool allows you to upload multilingual PDFs (Urdu, English, Bengali, and Chinese), 
        extract text (including from scanned PDFs), and ask questions based on the content.
        """
    )
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        # Save the uploaded PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        # Extract text from the uploaded PDF
        rag_system = RAGSystem(model)
        text = rag_system.extract_text_from_pdf("temp.pdf")
        st.text_area("Extracted Text", text, height=300)

        user_query = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            response = rag_system.answer_query(user_query, text)  # Pass the extracted text as context
            rag_system.add_to_memory(user_query, response)
            st.write("Response:", response)
            st.write("Chat History:", rag_system.get_memory())


if __name__ == "__main__":
    main()