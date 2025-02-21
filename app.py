import streamlit as st
import os
import tempfile
import re  # Import the regular expression module
from chromadb import PersistentClient, Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from groq import Groq
from utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_webpage, preprocess_text, get_embeddings

# LLM clients (replace with your actual API keys)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # For MVP

# Embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# ChromaDB initialization
temp_dir = tempfile.TemporaryDirectory()
db_client = PersistentClient(path=temp_dir.name)
collection = db_client.get_or_create_collection("cv_embeddings")

st.title("CV Automation System")

# Job Requirements Input
st.subheader("Job Requirements")
requirements_source = st.radio("Source:", ("File Upload", "Web Page Link", "Text Input"))

requirements_text = ""
if requirements_source == "File Upload":
    uploaded_file = st.file_uploader("Upload Job Requirements (PDF/DOCX)", type=["pdf", "docx"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            requirements_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            requirements_text = extract_text_from_docx(uploaded_file)
elif requirements_source == "Web Page Link":
    webpage_url = st.text_input("Enter Web Page URL")
    if webpage_url:
        requirements_text = extract_text_from_webpage(webpage_url)
elif requirements_source == "Text Input":
    requirements_text = st.text_area("Enter Job Requirements", height=200)

# CV Folder Input
st.subheader("CV Folder")
cv_folder = st.text_input("Enter Path to CV Folder")


def process_cvs(cv_folder, embedding_model, collection):
    cv_data = []
    for filename in os.listdir(cv_folder):
        if filename.endswith((".pdf", ".docx")):
            cv_path = os.path.join(cv_folder, filename)
            if filename.endswith(".pdf"):
                cv_text = extract_text_from_pdf(cv_path)
            else:  # .docx
                cv_text = extract_text_from_docx(cv_path)

            if cv_text:
                preprocessed_cv_text = preprocess_text(cv_text)
                cv_embedding = get_embeddings(preprocessed_cv_text, embedding_model)
                cv_data.append({"filename": filename, "text": preprocessed_cv_text, "embedding": cv_embedding})
                collection.add(embeddings=[cv_embedding], documents=[filename], ids=[filename])
    return cv_data


def get_llm_assessment(requirements_text, cv_text, llm_client):
    prompt = f"""
    Job Requirements: {requirements_text}

    Candidate CV: {cv_text}

    Assess the candidate's fit for the job.  Provide your assessment in the following format:

    ```
    Candidate Fit Assessment: <Candidate Name>
    Technical Skills: <List of skills>
    Experience: <Summary of experience>
    Education: <Education details>
    Areas of Improvement: <Areas where the candidate could improve>
    Conclusion: <Overall conclusion about the candidate's fit>
    Score: <High/Medium/Low>
    ```
    """

    response = llm_client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b",  # Or your chosen model
        messages=[{"role": "user", "content": prompt}]
    )
    assessment = response.choices[0].message.content
    return assessment


if st.button("Process CVs"):
    if requirements_text and cv_folder:
        preprocessed_requirements = preprocess_text(requirements_text)
        requirements_embedding = get_embeddings(preprocessed_requirements, embedding_model)

        cv_data = process_cvs(cv_folder, embedding_model, collection)

        results = []
        for cv_item in cv_data:
            results_db = collection.query(
                query_embeddings=[requirements_embedding],
                n_results=1
            )
            cv_text = cv_item["text"]
            llm_assessment = get_llm_assessment(requirements_text, cv_text, groq_client)  # or openai_client

            # Extract Score (High/Medium/Low) using regular expressions
            match = re.search(r"Score:\s*(High|Medium|Low)", llm_assessment, re.IGNORECASE)
            score = match.group(1) if match else "Unknown"  # Default if not found

            results.append({"filename": cv_item["filename"], "assessment": llm_assessment, "score": score})

        # Sort by score (High > Medium > Low > Unknown)
        score_order = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
        sorted_results = sorted(results, key=lambda x: score_order.get(x["score"], 3))  # Sort based on score_order

        # Structured output (table in Streamlit)
        st.subheader("Candidate Rankings")
        st.dataframe(
            [
                {
                    "Rank": i + 1,
                    "Filename": result["filename"],
                    "Score": result["score"],
                    "Assessment": result["assessment"],
                    "CV Link": os.path.join(cv_folder, result["filename"])  # Link to CV
                }
                for i, result in enumerate(sorted_results)
            ]
        )

    else:
        st.warning("Please provide both job requirements and CV folder path.")

# Cleanup code
st.session_state.temp_dir = temp_dir

def cleanup_temp_dir():
    if hasattr(st.session_state, "temp_dir"):
        st.session_state.temp_dir.cleanup()

st.session_state.on_close = cleanup_temp_dir