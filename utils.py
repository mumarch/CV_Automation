import PyPDF2
import docx
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}") # Changed to print for utils file
        return ""

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}") # Changed to print for utils file
        return ""

def extract_text_from_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator='\n')
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching webpage: {e}") # Changed to print for utils file
        return ""

def preprocess_text(text):
    # Basic cleaning (you'll likely want to expand this)
    text = text.lower()  # Lowercasing
    # Remove special characters, etc.
    return text

def get_embeddings(text, model):
    return model.encode(text)