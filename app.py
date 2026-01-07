import os
from flask import Flask, request, jsonify, render_template
from google import genai
import PyPDF2
from flask_cors import CORS

# ==============================
# CONFIG
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Replace with your actual Gemini API Key
client = genai.Client(api_key="")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# HOME ROUTE (UI)
# ==============================
@app.route("/")
def home():
    # Flask looks for this in the 'templates' folder
    return render_template("index.html")

# ==============================
# PDF PARSING
# ==============================
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

# ==============================
# LLM PROCESSING FUNCTIONS
# ==============================
def parse_resume(resume_text):
    prompt = f"""
    You are a professional resume parser.
    Extract the following from the text below:
    - Skills
    - Experience summary
    - Education
    - Tools & technologies
    
    Resume Text:
    {resume_text}
    
    Return the response in clear bullet points.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def parse_job_description(jd_text):
    prompt = f"""
    Extract the following from this Job Description:
    - Required skills
    - Responsibilities
    - Preferred qualifications
    
    Job Description:
    {jd_text}
    
    Return in clear bullet points.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def ats_match(parsed_resume, parsed_jd):
    prompt = f"""
    You are a high-end Applicant Tracking System (ATS).
    Compare the provided resume analysis against the job description analysis.
    
    Resume Analysis:
    {parsed_resume}
    
    Job Description Analysis:
    {parsed_jd}
    
    Provide a detailed report including:
    1. Match percentage (Format as: Match Score: XX%)
    2. Matching skills
    3. Critical missing skills
    4. Strengths of the candidate
    5. Specific improvement suggestions for this JD
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ==============================
# API ROUTE (PDF UPLOAD & ANALYSIS)
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    # Save PDF locally
    filename = resume_file.filename
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    resume_file.save(pdf_path)

    try:
        # 1. Extract text
        resume_text = extract_text_from_pdf(pdf_path)
        if not resume_text.strip():
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # 2. Parse components using Gemini
        parsed_resume = parse_resume(resume_text)
        parsed_jd = parse_job_description(jd_text)

        # 3. Perform ATS Matching
        ats_result = ats_match(parsed_resume, parsed_jd)

        return jsonify({
            "parsed_resume": parsed_resume,
            "parsed_job_description": parsed_jd,
            "ats_result": ats_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Optional: Clean up file after processing
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    # Running on 8080 as requested

    app.run(debug=True, port=8080)
