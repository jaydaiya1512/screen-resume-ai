from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Function to extract text from PDFs
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Function to rank resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_vector], resume_vectors)[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    job_description = request.form.get("job_description")
    files = request.files.getlist("resumes")

    if not job_description or not files:
        return jsonify({"error": "Missing input"}), 400

    resumes = [extract_text_from_pdf(file) for file in files]
    scores = rank_resumes(job_description, resumes)

    results = [{"resume": file.filename, "score": score} for file, score in zip(files, scores)]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)

