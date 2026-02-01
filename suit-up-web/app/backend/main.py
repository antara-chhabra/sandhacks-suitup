"""
Unified SuitUp Backend - Resume, Interview Questions, Visual/Speech Analysis
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import json
import re
import subprocess
import threading
import time
import base64
import uuid
from pathlib import Path
from datetime import datetime

# Paths (suit-up-web/app/backend; extras at repo root)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
SESSIONS_DIR = BASE_DIR / "sessions"
SPEECH_INTENT_DIR = PROJECT_ROOT / "extras" / "speech_intent" if (PROJECT_ROOT / "extras" / "speech_intent").exists() else PROJECT_ROOT / "speech_intent"
VISUAL_PERCEPTION_DIR = PROJECT_ROOT / "suit-up-web" / "visual_perception"
INTERVIEWQS_DIR = PROJECT_ROOT / "extras" / "interviewqs_comppos" if (PROJECT_ROOT / "extras" / "interviewqs_comppos").exists() else PROJECT_ROOT / "interviewqs_comppos"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ LLM Setup ============
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key")
DEMO_MODE = GROQ_API_KEY in (None, "", "your_groq_api_key")
llm = None
if not DEMO_MODE:
    try:
        from llama_index.llms.groq import Groq
        llm = Groq(model="llama3-70b-8192")
    except Exception:
        DEMO_MODE = True

# In-memory session storage (use Redis in production)
sessions: dict = {}
visual_analyzers: dict = {}
speech_results: dict = {}
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


# ============ Request Models ============
class CompanyPositionRequest(BaseModel):
    company: str
    position: str


class StartInterviewRequest(BaseModel):
    resume_questions: list
    company_questions: dict
    company: str
    position: str


class AudioChunkRequest(BaseModel):
    session_id: str
    audio_base64: str


class FrameRequest(BaseModel):
    session_id: str
    frame_base64: str


class EndInterviewRequest(BaseModel):
    session_id: str
    conversation: list  # [{role, content}, ...]


# ============ Resume Processing ============
@app.post("/process-resume")
async def process_resume(file: UploadFile = File(...)):
    file_path = UPLOADS_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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

    from llama_index.core import SimpleDirectoryReader, SummaryIndex
    reader = SimpleDirectoryReader(input_files=[str(file_path)])
    documents = reader.load_data()
    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine(llm=llm)
    prompt = (
        "Based on this resume, generate 5 challenging behavioral interview questions "
        "tailored to the candidate's experience. Return them as a numbered list. "
        "One question per line, no numbering."
    )
    response = query_engine.query(prompt)
    questions = [q.strip() for q in str(response).strip().split("\n") if q.strip()]
    if not questions:
        questions = ["Tell me about yourself.", "Describe a challenging project."]

    return {"questions": questions, "filename": file.filename, "demo": False}


# ============ Company/Position Questions (interviewqs_comppos) ============
def run_interviewqs(company: str, position: str) -> dict:
    """Run interview question generator with company/position."""
    try:
        import sys
        sys.path.insert(0, str(INTERVIEWQS_DIR))
        from interview_api import generate_interview_questions
        out_path = SESSIONS_DIR / f"company_q_{uuid.uuid4().hex[:8]}.json"
        return generate_interview_questions(company, position, str(out_path))
    except Exception as e:
        print(f"interviewqs error: {e}")
        return _demo_company_questions(company, position)


def _demo_company_questions(company: str, position: str) -> dict:
    return {
        "company_values": [f"Excellence at {company}", "Collaboration", "Innovation"],
        "technical_questions": [
            f"Describe a technical challenge you solved.",
            f"What experience do you have with {position}?",
            "Explain a system you designed."
        ],
        "behavioral_questions": [
            f"Why do you want to join {company}?",
            "Tell me about a time you faced a difficult teammate.",
            "How do you handle ambiguity?"
        ]
    }


@app.post("/generate-company-questions")
async def generate_company_questions(req: CompanyPositionRequest):
    company = req.company.strip() or "Company"
    position = req.position.strip() or "Role"
    data = run_interviewqs(company, position)
    # Also save to a txt for the AI interviewer
    out_path = SESSIONS_DIR / f"company_questions_{uuid.uuid4().hex[:8]}.json"
    with open(out_path, "w") as f:
        json.dump({"company": company, "position": position, **data}, f, indent=2)
    return {"company": company, "position": position, **data}


# ============ Visual Perception (body_language_detection) ============
class _StubVisualAnalyzer:
    """Fallback when full emotion model is not available; still counts frames for report."""

    def __init__(self):
        self.recording = False
        self.start_time = None
        self.metrics = {"total_frames": 0}

    def start_recording(self):
        self.recording = True
        self.start_time = __import__("time").time()

    def analyze_frame(self, frame):
        if self.recording:
            self.metrics["total_frames"] += 1
        return frame, None

    def stop_recording(self):
        self.recording = False
        if self.metrics["total_frames"] == 0:
            return {"error": "No data recorded"}
        return {
            "overall_score": 0,
            "duration_seconds": (__import__("time").time() - self.start_time) if self.start_time else 0,
            "total_frames_analyzed": self.metrics["total_frames"],
            "eye_contact": {"percentage": 0, "score": 0},
            "emotions": {"most_common": "N/A", "positive_percentage": 0, "negative_percentage": 0, "neutral_percentage": 0},
            "recommendations": [{"category": "Visual", "severity": "low", "message": "Emotion model was unavailable; only frame count recorded.", "tip": "Install TensorFlow and emotion_model.h5 for full analysis."}],
        }


def init_visual_analyzer(session_id: str):
    """Initialize visual analyzer for a session; fallback to stub if model fails."""
    model_path = VISUAL_PERCEPTION_DIR / "emotion_model.h5"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "suit-up-web" / "visual_perception" / "emotion_model.h5"
    try:
        import sys
        sys.path.insert(0, str(VISUAL_PERCEPTION_DIR))
        from body_language_detection import InterviewAnalyzer
        analyzer = InterviewAnalyzer(str(model_path) if model_path.exists() else None)
        analyzer.start_recording()
        visual_analyzers[session_id] = analyzer
        return True
    except Exception as e:
        print(f"Visual analyzer init error (using stub): {e}")
        stub = _StubVisualAnalyzer()
        stub.start_recording()
        visual_analyzers[session_id] = stub
        return True


@app.post("/process-frame")
async def process_frame(req: FrameRequest):
    """Process a single video frame for body language analysis."""
    if req.session_id not in visual_analyzers:
        init_visual_analyzer(req.session_id)
    analyzer = visual_analyzers.get(req.session_id)
    if not analyzer:
        return {"status": "skipped", "reason": "analyzer not ready"}

    try:
        img_data = base64.b64decode(req.frame_base64)
        import numpy as np
        import cv2
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "error", "reason": "invalid frame"}
        _, analysis = analyzer.analyze_frame(frame)
        return {"status": "ok", "analysis": analysis}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def stop_visual_analyzer(session_id: str) -> dict | None:
    """Stop analyzer and return report."""
    analyzer = visual_analyzers.pop(session_id, None)
    if not analyzer:
        return None
    report = analyzer.stop_recording()
    return report


# ============ Speech Intent (Whisper + ML) ============
def process_audio_chunk(session_id: str, audio_b64: str) -> dict:
    """Process audio with Whisper + features, save to session."""
    try:
        import numpy as np
        import tempfile
        import sys
        sys.path.insert(0, str(SPEECH_INTENT_DIR))
        import joblib
        from features import extract_audio_features

        audio_bytes = base64.b64decode(audio_b64)
        suffix = ".webm"  # MediaRecorder sends webm
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = get_whisper_model()
        result = model.transcribe(tmp_path)
        transcription = (result.get("text") or "").strip()

        try:
            feats = extract_audio_features(tmp_path)
            from features import pad_or_truncate
            features = pad_or_truncate(feats, 88)
        except Exception as fe:
            print(f"Feature extraction skipped: {fe}")
            features = np.zeros(88)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        VOCAL_STATES = ["nervous", "confident", "overconfident", "monotone"]
        clf_path = SPEECH_INTENT_DIR / "clf.pkl"
        state = "unknown"
        if clf_path.exists():
            clf = joblib.load(clf_path)
            pred = clf.predict(features.reshape(1, -1))[0]
            state = VOCAL_STATES[pred] if pred < len(VOCAL_STATES) else "unknown"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "transcription": transcription,
            "predicted_state": state,
            "features": features.tolist() if hasattr(features, "tolist") else list(features)
        }
        if session_id not in speech_results:
            speech_results[session_id] = []
        speech_results[session_id].append(entry)
        return {"transcription": transcription, "state": state}
    except Exception as e:
        print(f"Speech processing error: {e}")
        return {"transcription": "", "state": "unknown", "error": str(e)}


@app.post("/process-audio")
async def process_audio(req: AudioChunkRequest):
    return process_audio_chunk(req.session_id, req.audio_base64)


class TranscribeRequest(BaseModel):
    audio_base64: str
    session_id: str | None = None


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    """Whisper-only transcription for interview flow. Optionally saves to session."""
    try:
        import tempfile
        audio_bytes = base64.b64decode(req.audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            model = get_whisper_model()
            result = model.transcribe(tmp_path)
            text = (result.get("text") or "").strip()
            if req.session_id and text:
                if req.session_id not in speech_results:
                    speech_results[req.session_id] = []
                speech_results[req.session_id].append({
                    "timestamp": datetime.now().isoformat(),
                    "transcription": text,
                    "predicted_state": "unknown",
                })
            return {"transcription": text}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        print(f"Transcribe error: {e}")
        return {"transcription": "", "error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ============ Interview Session Lifecycle ============
@app.post("/start-interview")
async def start_interview(req: StartInterviewRequest):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "resume_questions": req.resume_questions,
        "company_questions": req.company_questions,
        "company": req.company,
        "position": req.position,
        "started_at": datetime.now().isoformat(),
        "conversation": []
    }
    init_visual_analyzer(session_id)
    return {"session_id": session_id}


@app.post("/end-interview")
async def end_interview(req: EndInterviewRequest):
    session_id = req.session_id
    if session_id not in sessions:
        sessions[session_id] = {"conversation": [], "company": "N/A", "position": "N/A"}

    sessions[session_id]["conversation"] = req.conversation
    sessions[session_id]["ended_at"] = datetime.now().isoformat()

    # 1. Stop visual analyzer and get report
    visual_report = stop_visual_analyzer(session_id)

    # 2. Get speech results
    voice_entries = speech_results.pop(session_id, [])

    # 3. Llama content feedback (only if we have user input)
    conv = req.conversation or []
    user_msgs = [m for m in conv if m.get("role") == "user" and (m.get("content") or "").strip()]
    content_feedback = await _get_llama_content_feedback(req.conversation) if user_msgs else ""

    # 4. Build combined feedback
    feedback = _build_feedback_report(
        session_id, sessions[session_id],
        visual_report, voice_entries, content_feedback
    )
    sessions[session_id]["feedback"] = feedback

    return {"session_id": session_id, "feedback": feedback}


async def _get_llama_content_feedback(conversation: list) -> str:
    """Use Ollama to evaluate interview answers based on actual user responses."""
    transcript = "\n".join([f"{m.get('role','')}: {m.get('content','')}" for m in conversation if m.get("role") != "system"])
    user_answers = [m.get("content", "") for m in conversation if m.get("role") == "user" and m.get("content")]
    prompt = f"""You are an expert interview coach. Evaluate this mock interview transcript. Your feedback must be specific to what the candidate actually said — reference their exact answers where relevant. Do not give generic advice.

Transcript:
{transcript[:5000]}

Candidate's answers (for reference): {chr(10).join([a[:200] for a in user_answers[:15]])}

Provide 3-5 bullet points of constructive feedback:
- Reference specific things the candidate said (e.g. "When you said X, consider adding Y").
- Content quality and relevance to the questions asked.
- Clarity and structure of their answers.
- One or two concrete areas to improve, tied to their actual responses.
- Strengths they demonstrated.

Keep each point under 2 lines. Be encouraging, professional, and specific to this transcript. No generic phrases."""

    try:
        import ollama
        r = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return r.get("message", {}).get("content", "Content evaluation unavailable.")
    except Exception as e:
        print(f"Llama content feedback error: {e}")
        return "Content evaluation skipped (Ollama not running)."


def _build_feedback_report(session_id: str, session: dict, visual_report: dict | None, voice_entries: list, content_feedback: str) -> dict:
    """Build combined feedback in interview_report.txt format."""
    company = (session.get("company") or "N/A").strip()
    position = (session.get("position") or "N/A").strip()

    # No user input: no transcriptions, no user messages in conversation
    conv = session.get("conversation") or []
    user_messages = [m for m in conv if (m.get("role") == "user" and (m.get("content") or "").strip())]
    has_transcriptions = any((e.get("transcription") or "").strip() for e in voice_entries)
    has_visual_data = visual_report and (visual_report.get("total_frames_analyzed") or 0) > 0
    if not user_messages and not has_transcriptions and not has_visual_data:
        report_text = (
            "INTERVIEW PERFORMANCE REPORT\n"
            + "=" * 60 + "\n\n"
            + f"Company: {company}\n"
            + f"Position: {position}\n\n"
            + "Feedback report unavailable. No input found.\n\n"
            + "=" * 60
        )
        report_path = SESSIONS_DIR / f"interview_report_{session_id[:8]}.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        return {"report_text": report_text, "report_path": str(report_path)}

    lines = [
        "INTERVIEW PERFORMANCE REPORT",
        "=" * 60,
        "",
        f"Company: {company}",
        f"Position: {position}",
        "",
    ]

    if visual_report and not visual_report.get("error"):
        lines.extend([
            f"Overall Score: {visual_report.get('overall_score', 0):.1f}/100",
            f"Duration: {visual_report.get('duration_seconds', 0):.1f} seconds",
            f"Frames Analyzed: {visual_report.get('total_frames_analyzed', 0)}",
            "",
            "-" * 60,
            "EYE CONTACT ANALYSIS",
            "-" * 60,
            f"Percentage: {visual_report.get('eye_contact', {}).get('percentage', 0):.1f}%",
            f"Score: {visual_report.get('eye_contact', {}).get('score', 0):.1f}/100",
            "",
            "-" * 60,
            "EMOTION ANALYSIS",
            "-" * 60,
            f"Most Common: {visual_report.get('emotions', {}).get('most_common', 'N/A')}",
            f"Positive: {visual_report.get('emotions', {}).get('positive_percentage', 0):.1f}%",
            f"Negative: {visual_report.get('emotions', {}).get('negative_percentage', 0):.1f}%",
            f"Neutral: {visual_report.get('emotions', {}).get('neutral_percentage', 0):.1f}%",
            "",
            "-" * 60,
            "RECOMMENDATIONS",
            "-" * 60,
        ])
        for i, rec in enumerate(visual_report.get("recommendations", []), 1):
            lines.append(f"\n{i}. [{rec.get('severity', 'low').upper()}] {rec.get('category', '')}")
            lines.append(f"   {rec.get('message', '')}")
            lines.append(f"   Tip: {rec.get('tip', '')}")
    elif visual_report and visual_report.get("error"):
        lines.extend([
            "",
            "-" * 60,
            "VISUAL / EMOTION ANALYSIS",
            "-" * 60,
            "No frames were received or the emotion model was unavailable during the interview.",
            "Ensure your camera is on and shared for the next session.",
            "",
        ])
    else:
        lines.extend([
            "",
            "-" * 60,
            "VISUAL / EMOTION ANALYSIS",
            "-" * 60,
            "Visual analyzer was not initialized (camera may not have been shared).",
            "",
        ])

    if voice_entries:
        lines.extend([
            "",
            "-" * 60,
            "VOICE / SPEECH ANALYSIS",
            "-" * 60,
        ])
        for e in voice_entries[-10:]:
            trans = (e.get("transcription") or "").strip()
            if not trans:
                continue
            state = e.get("predicted_state", "unknown")
            lines.append(f"• [{state}] {trans[:150]}")
        lines.append("")

    lines.extend([
        "-" * 60,
        "CONTENT FEEDBACK (AI Evaluation)",
        "-" * 60,
        content_feedback or "N/A",
        "",
        "=" * 60
    ])

    report_text = "\n".join(lines)
    report_path = SESSIONS_DIR / f"interview_report_{session_id[:8]}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    return {"report_text": report_text, "report_path": str(report_path)}


@app.get("/get-feedback/{session_id}")
async def get_feedback(session_id: str):
    if session_id not in sessions or "feedback" not in sessions[session_id]:
        return {"ready": False}
    return {"ready": True, "feedback": sessions[session_id]["feedback"]}
