import gradio as gr
from sentence_transformers import SentenceTransformer, util

# Load Hugging Face model
model = SentenceTransformer(
    "anass1209/resume-job-matcher-all-MiniLM-L6-v2"
)

def analyze_resume(resume, job):

    if not resume or not job:
        return "Please enter both the resume and job description."

    resume_embedding = model.encode(resume)
    job_embedding = model.encode(job)

    similarity = util.cos_sim(
        resume_embedding,
        job_embedding
    ).item()

    score = round(similarity * 100, 2)

    if score >= 70:
        result = "Strong match"
    elif score >= 50:
        result = "Moderate match"
    else:
        result = "Low match"

    return f"""
Similarity Score: {score}%

Result: {result}

This score represents semantic similarity between
the resume and job description.
"""

demo = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.Textbox(
            lines=10,
            label="Resume"
        ),
        gr.Textbox(
            lines=10,
            label="Job Description"
        )
    ],
    outputs=gr.Textbox(
        lines=8,
        label="Analysis"
    ),
    title="AI Resume Analyzer",
    description="Compare a resume with a job description using a Hugging Face model."
)

demo.launch()