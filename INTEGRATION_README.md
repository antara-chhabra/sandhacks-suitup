# SuitUp - Integrated Interview Practice Platform

## Flow

1. **Login** (suit-up-web) → Enter username/password
2. **Resume Upload** → Upload PDF, backend processes with Llama, saves behavioral questions
3. **Company & Position** → Enter company and role; runs interviewqs (Llama) for tailored questions
4. **Fit Check** → Camera + mic test before interview
5. **Interview Session** → AI interviewer (Barney), webcam, chat with Llama, visual + speech analysis in background
6. **Feedback Form** → Scrollable report (visual perception + speech intent + content evaluation)
7. **Thank You** → Retry (back to company/position) or Exit

## Running the App

### Prerequisites

- Node.js 18+
- Python 3.10+
- [Ollama](https://ollama.ai) with `llama3`: `ollama run llama3`
- (Optional) GROQ_API_KEY for resume processing; otherwise demo mode

### Terminal 1: Python Backend (port 8000)

```bash
cd suit-up-web
npm run backend
# Or: cd suit-up-web/app/backend && uvicorn main:app --reload --port 8000
```

### Terminal 2: Next.js Frontend (port 3000)

```bash
cd suit-up-web
npm run dev
```

### Terminal 3: Ollama (for chat + interview questions)

```bash
ollama serve   # if not already running
ollama run llama3
```

### Optional: GROQ for Resume (faster)

```bash
export GROQ_API_KEY=your_key
```

## Architecture

- **Frontend**: suit-up-web (Next.js, Tailwind, navy/gold theme)
- **Backend**: FastAPI at `localhost:8000`
  - `/process-resume` – PDF → Llama → behavioral questions
  - `/generate-company-questions` – company/position → interviewqs (Ollama) → questions
  - `/start-interview` – create session, init visual analyzer
  - `/process-frame` – receive webcam frames for body language analysis
  - `/process-audio` – receive audio for Whisper + vocal state (speech_intent)
  - `/end-interview` – generate combined report (visual + speech + Llama content feedback)

- **Chat**: Next.js `/api/chat` → Ollama (llama3)

## Speech Intent

The tkinter GUI is replaced by integration in the interview session:
- Audio is captured via MediaRecorder during the interview
- Sent to `/process-audio` which runs Whisper + ML classifier (clf.pkl)
- Results appear in the feedback report under "Voice / Speech Analysis"

## Visual Perception

- `body_language_detection.py` runs on frames sent from the browser
- Uses `emotion_model.h5` for facial expressions
- Produces eye contact, emotions, recommendations in the feedback

## File Structure

```
suit-up-web/
├── app/
│   ├── api/chat/route.ts     # Ollama proxy
│   ├── backend/main.py       # Unified Python API
│   ├── company_position/     # Company & role input
│   ├── feedback/             # Report + retry/exit
│   ├── interview_session/    # AI interview + Barney
│   ├── interview_setup/      # Camera/mic check
│   ├── login/
│   ├── thankyou/
│   └── upload_resume/
├── visual_perception/        # emotion_model.h5, body_language_detection.py
speech_intent/                # features.py, clf.pkl (used by backend)
interviewqs_comppos/          # interview_api.py, interview_scraper.py
```
