from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

import os
os.makedirs("uploads", exist_ok=True)

app = FastAPI()

# Enable CORS so your React app can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Llama (Using Groq is free and very fast for hackathons)
# Read GROQ key from env. If not set, use demo mode to avoid LLM calls during local dev.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key")
DEMO_MODE = GROQ_API_KEY in (None, "", "your_groq_api_key")

# Lazy LLM initialization (only if not demo)
llm = None
if not DEMO_MODE:
    try:
        from llama_index.llms.groq import Groq
        llm = Groq(model="llama3-70b-8192")
    except Exception:
        # If LLM libs fail to import, fall back to demo mode
        DEMO_MODE = True

@app.post("/process-resume")
async def process_resume(file: UploadFile = File(...)):
    # 1. Save file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # If demo mode is active (no API key or LLM failed to load), return sample questions
    if DEMO_MODE:
        return {
            "questions": [
                "Tell me about a time you faced a major technical challenge.",
                "Describe a project where you took ownership end-to-end.",
                "How do you prioritize competing deadlines and stakeholder requests?",
                "Tell me about a time you received difficult feedback and how you handled it.",
                "Describe a situation where you had to learn a new technology quickly."
            ],
            "filename": file.filename,
            "demo": True
        }

    # 2. Read PDF & Generate Questions using llama_index + Groq
    from llama_index.core import SimpleDirectoryReader, SummaryIndex
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    
    # 3. Ask Llama to generate specific questions
    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine(llm=llm)
    
    prompt = (
        "Based on this resume, generate 5 challenging behavioral interview questions "
        "tailored to the candidate's experience. Return them as a numbered list."
    )
    
    response = query_engine.query(prompt)
    
    return {
        "questions": str(response).split("\n"),
        "filename": file.filename,
        "demo": False
    }