# SuitUp – Your Interview Companion

**SuitUp** is an AI-powered interview coach that helps you practice for job interviews in a realistic, controlled environment. It doesn’t just ask questions—it evaluates your facial expressions, tone of voice, and answers, then gives you actionable feedback.

---

## The Problem

Interviews are tough. You can memorize common questions, read every tip online, and rehearse answers—but that doesn’t fully prepare you for the moment when someone is on the other end and your mind goes blank.

Most of us use AI models, find questions from all over the web, and generate Q&A lists. **But how do you actually prep in a controlled environment?**

- You don’t know who will be on the other end in a real interview.
- You need practice with a virtual interviewer and real-time feedback.
- You need to know where to focus: body language, speech, and content.

---

## What SuitUp Does

SuitUp is a **virtual interview companion** that:

1. **Acts as your interviewer** – A virtual interviewer (e.g. Barney) introduces himself and asks resume-based and company-specific questions.
2. **Uses your resume** – Upload your resume; the system generates tailored questions so the interview feels real (crucial for entry-level and career switchers).
3. **Uses company and role** – You enter company and position; we use web scraping and AI to pull interview questions and values from the web.
4. **Analyzes you in real time** –  
   - **Facial:** Emotion model (expressions, head movement, nodding).  
   - **Voice:** Whisper for transcription; pitch, tone, pauses, stutters.  
   - **Content:** Llama evaluates your answers and gives feedback tied to what you said.
5. **Gives you a report** – A scrollable feedback report with company and position (e.g. **Google**, **Software Engineer**), scores, and improvement suggestions. No fake evaluations when there’s no input—you’ll see *“Feedback report unavailable. No input found.”* when nothing was captured.

---

## Key Features

| Feature | Description |
|--------|-------------|
| **Virtual interviewer** | AI-driven interviewer; optional avatar (e.g. deepfake-style) so you don’t know “who” you’ll get. |
| **Resume-aware questions** | Upload a PDF; AI generates questions from your experience. |
| **Company-specific questions** | Web scraping + Llama for company values and common interview questions. |
| **Body language & speech** | Facial expressions, nodding, eye contact; voice tone, pitch, pauses. |
| **Scoring & feedback** | Speech confidence, clarity/rambling, body language, overall readiness. |
| **Personalized suggestions** | Tips based on your session, not generic advice. |

---

## Technologies

| Layer | Tech |
|-------|------|
| **Frontend** | Next.js, React, Tailwind CSS |
| **Backend** | Python, FastAPI |
| **AI / conversation** | Llama (Ollama) for interview flow and content feedback |
| **Voice** | Whisper (transcription), librosa (features), optional ML classifier |
| **Visual** | OpenCV, Keras emotion model (facial expressions, nodding, eye contact) |
| **Company questions** | Web scraping (e.g. trafilatura, DuckDuckGo) + Llama |

---

## Project Structure

```
.
├── suit-up-web/       # Frontend (Next.js): login, resume upload, company/position, interview, feedback
├── backend/          # Backend (FastAPI): resume, company questions, visual/speech, Llama
│   ├── main.py
│   └── visual_perception/
├── extras/            # Supporting modules (interviewqs, speech_intent, scripts, etc.)
└── README.md
```

---

## Quick Start

**Prerequisites:** Node.js, Python 3, Ollama (for Llama).

### 1. Ollama (Llama)

```bash
ollama serve
ollama run llama3
```

### 2. Backend (port 8000)

From repo root:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Or from `suit-up-web`:

```bash
npm install
cd suit-up-web
npm run backend
```

### 3. Frontend (port 3000)

```bash
cd suit-up-web
npm install
npm run dev
```

Open **http://localhost:3000**.

---

## User Flow

1. **Login** → **Resume upload** (PDF; you can remove and re-upload).
2. **Company & position** (e.g. Google, Software Engineer).
3. **Interview setup** → Camera and mic check.
4. **Interview** – Virtual interviewer says *“My name is Barney, and I will be your interviewer today.”* You toggle the mic on to speak and off when done; your answers are transcribed and shown in the chat.
5. **End call** (or 40 min) → **Feedback report** (scrollable), with company and position and *“Feedback report unavailable. No input found.”* when no input was captured.
6. **Thank you & exit** or **Retry** (back to company/position).

---

## Hackathon Fit

- **Best Overall Hack** – Full-stack AI, real-world problem.
- **UCSD Basement** – AI as a human-like interviewer.
- **EyePop.ai** – Can be extended with EyePop for gaze/facial analysis.

---

## Inspiration

We asked: *What if people could practice interviews in a controlled, safe environment—and get honest feedback on how they’re actually performing?*

SuitUp is that: a brutally honest mentor in your laptop, but friendly. More Barney than boss.

---

## License

Built for hackathon demo. Use and extend as you like.

**Suit up.**
**Team SAAS**
