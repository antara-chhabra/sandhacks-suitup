"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Mic, MicOff, PhoneOff, Loader2 } from "lucide-react";

const BACKEND = typeof process !== "undefined" && process.env.NEXT_PUBLIC_BACKEND_URL
  ? process.env.NEXT_PUBLIC_BACKEND_URL
  : "http://localhost:8000";
const INTERVIEW_DURATION_MS = 40 * 60 * 1000;
const FRAME_INTERVAL_MS = 2000;

export default function InterviewSessionPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<{ role: string; content: string; isThinking?: boolean }[]>([]);
  const messagesRef = useRef<{ role: string; content: string; isThinking?: boolean }[]>([]);
  messagesRef.current = messages;
  const [isStarted, setIsStarted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [isEnding, setIsEnding] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const handleReplyRef = useRef<((text: string) => Promise<void>) | null>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Attach camera stream to video when interview view is mounted
  useEffect(() => {
    if (!isStarted || !videoRef.current || !streamRef.current) return;
    videoRef.current.srcObject = streamRef.current;
  }, [isStarted]);

  const buildSystemPrompt = useCallback(() => {
    const resumeQs = JSON.parse(localStorage.getItem("interview_questions") || "[]");
    const companyData = JSON.parse(localStorage.getItem("company_questions") || "{}");
    const company = localStorage.getItem("company") || "the company";
    const position = localStorage.getItem("position") || "the role";
    const resumeList = Array.isArray(resumeQs) ? resumeQs : [resumeQs];
    const techQs = companyData.technical_questions || [];
    const behavQs = companyData.behavioral_questions || [];
    const allQuestions = [
      ...resumeList.slice(0, 5),
      ...techQs.slice(0, 3),
      ...behavQs.slice(0, 3),
    ].filter(Boolean);
    return `You are a professional hiring manager conducting an interview at ${company} for the ${position} role. Your name is Barney. At the very start of the interview, introduce yourself by saying: "My name is Barney, and I will be your interviewer today." Be courteous and business-appropriate. Ask ONE question at a time. Use these themes: ${allQuestions.slice(0, 8).join(" | ")}. After the candidate answers, briefly acknowledge or give one sentence of constructive feedback, then ask the next question. Keep responses concise and professional. Wrap up after 5-7 exchanges.`;
  }, []);

  const speakWithSync = useCallback((text: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    const maleVoice = voices.length
      ? (voices.find((v) => v.name.toLowerCase().includes("male"))
          || voices.find((v) => v.lang.startsWith("en") && v.name.includes("Male"))
          || voices.find((v) => v.lang.includes("en-GB"))
          || voices[0])
      : null;
    if (maleVoice) utterance.voice = maleVoice;
    utterance.rate = 0.95;
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
    utterance.onboundary = (event) => {
      if (event.name === "word") {
        const idx = event.charIndex;
        const rest = text.slice(idx);
        const word = rest.split(/\s+/)[0] || "";
        const spokenPart = text.slice(0, idx + word.length);
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          return [...prev.slice(0, -1), { ...last, content: spokenPart }];
        });
      }
    };
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => {
      setIsSpeaking(false);
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        return [...prev.slice(0, -1), { ...last, content: text }];
      });
    };
    window.speechSynthesis.speak(utterance);
  }, []);

  const getAIResponse = useCallback(
    async (history: { role: string; content: string }[]) => {
      setMessages((prev) => [...prev, { role: "assistant", content: "Thinking...", isThinking: true }]);
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: history }),
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.error || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const aiText = data.message?.content || "I'm having trouble connecting. Let's try again.";
        setMessages((prev) => prev.filter((m) => !m.isThinking));
        speakWithSync(aiText);
      } catch (e) {
        setMessages((prev) => prev.filter((m) => !m.isThinking));
        const msg = e instanceof Error ? e.message : "Connection failed";
        speakWithSync(`Sorry, I couldn't connect to the AI. ${msg}`);
      }
    },
    [speakWithSync]
  );

  const handleReply = useCallback(
    async (text: string) => {
      if (!text || !text.trim()) return;
      const current = messagesRef.current.filter((m) => m.role !== "system");
      const newHistory = [
        { role: "system", content: buildSystemPrompt() },
        ...current,
        { role: "user", content: text.trim() },
      ];
      setMessages((prev) => [...prev.filter((m) => m.role !== "system"), { role: "user", content: text.trim() }]);
      await getAIResponse(newHistory);
    },
    [getAIResponse, buildSystemPrompt]
  );
  handleReplyRef.current = handleReply;

  // Helper to find valid MIME type for the browser
  const getSupportedMimeType = useCallback(() => {
    const types = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
      "audio/ogg",
    ];
    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return ""; // Empty lets browser choose default
  }, []);

  const stopRecordingAndTranscribe = useCallback(async () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state !== "recording") return;
    setIsRecording(false);
    setIsTranscribing(true);

    try {
      recorder.stop();
    } catch (_) {}

    await new Promise<void>((resolve) => {
      const onStop = () => {
        recorder.removeEventListener("stop", onStop);
        resolve();
      };
      recorder.addEventListener("stop", onStop);
    });

    const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
    audioChunksRef.current = [];

    if (blob.size < 200) {
      setIsTranscribing(false);
      mediaRecorderRef.current = null;
      return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
      const b64 = (reader.result as string).split(",")[1];
      if (!b64) {
        setIsTranscribing(false);
        mediaRecorderRef.current = null;
        return;
      }
      try {
        const res = await fetch(`${BACKEND}/transcribe`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audio_base64: b64,
            session_id: sessionIdRef.current || undefined,
          }),
        });
        const data = await res.json();
        const transcript = (data.transcription ?? "").toString().trim();
        if (handleReplyRef.current) {
          if (transcript) {
            await handleReplyRef.current(transcript);
          } else {
            setMessages((prev) => [...prev.filter((m) => m.role !== "system"), { role: "user", content: "(No speech detected — try again)" }]);
          }
        }
      } catch (e) {
        console.error("Transcribe error:", e);
        setBackendError("Could not transcribe. Is the backend running on port 8000?");
        setMessages((prev) => [...prev.filter((m) => m.role !== "system"), { role: "user", content: "(Transcription failed — check backend)" }]);
      } finally {
        setIsTranscribing(false);
        setIsRecording(false);
        mediaRecorderRef.current = null;
      }
    };
    reader.readAsDataURL(blob);
  }, []);

  const toggleRecording = useCallback(() => {
    if (isTranscribing || isSpeaking) return;

    if (isRecording) {
      stopRecordingAndTranscribe();
    } else {
      const stream = streamRef.current;
      if (!stream || !stream.active || stream.getAudioTracks().length === 0) {
        console.error("Stream invalid or missing audio tracks");
        return;
      }

      try {
        const mimeType = getSupportedMimeType();
        // Removed audioBitsPerSecond to fix NotSupportedError
        const options: MediaRecorderOptions = mimeType ? { mimeType } : {};
        
        const recorder = new MediaRecorder(stream, options);
        audioChunksRef.current = [];

        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
        };
        
        mediaRecorderRef.current = recorder;
        recorder.start(1000); // Record in 1s chunks
        setIsRecording(true);
      } catch (err) {
        console.error("Recorder start failed:", err);
        // Fallback to absolute defaults
        try {
            const fallback = new MediaRecorder(stream);
            audioChunksRef.current = [];
            fallback.ondataavailable = e => { if (e.data.size>0) audioChunksRef.current.push(e.data); };
            mediaRecorderRef.current = fallback;
            fallback.start(1000);
            setIsRecording(true);
        } catch(e2) {
            console.error("Fallback failed:", e2);
            setBackendError("Recording not supported by this browser.");
        }
      }
    }
  }, [isRecording, isTranscribing, isSpeaking, stopRecordingAndTranscribe, getSupportedMimeType]);

  const startInterview = useCallback(async () => {
    setBackendError(null);
    const resumeQuestions = JSON.parse(localStorage.getItem("interview_questions") || "[]");
    const companyQuestions = JSON.parse(localStorage.getItem("company_questions") || "{}");
    const company = localStorage.getItem("company") || "Company";
    const position = localStorage.getItem("position") || "Role";

    // Start camera first
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: 15 },
        audio: true,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err) {
      console.error("Media error:", err);
      setBackendError("Could not access camera/microphone. Check permissions.");
    }

    try {
      const res = await fetch(`${BACKEND}/start-interview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume_questions: Array.isArray(resumeQuestions) ? resumeQuestions : [resumeQuestions],
          company_questions: companyQuestions,
          company,
          position,
        }),
      });
      if (!res.ok) throw new Error("Backend unavailable");
      const data = await res.json();
      const sid = data.session_id;
      setSessionId(sid);
      sessionIdRef.current = sid;
    } catch (e) {
      console.error("Start interview error:", e);
      setBackendError("Backend may be offline. Interview will continue with limited features.");
      const sid = "local-" + Date.now();
      setSessionId(sid);
      sessionIdRef.current = sid;
    }

    const system = { role: "system", content: buildSystemPrompt() };
    setMessages([system]);
    setIsStarted(true);

    // Initial AI greeting
    await getAIResponse([system, { role: "user", content: "Start the interview." }]);
  }, [buildSystemPrompt, getAIResponse]);

  // Frame capture logic
  useEffect(() => {
    if (!isStarted || !sessionId) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const sid = sessionId;
    const sendFrame = () => {
      if (!videoRef.current || !canvasRef.current) return;
      const v = videoRef.current;
      const c = canvasRef.current;
      if (v.readyState < 1) return;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      try {
        c.width = 160;
        c.height = 120;
        ctx.drawImage(v, 0, 0, 160, 120);
        c.toBlob(
          (blob) => {
            if (!blob) return;
            const reader = new FileReader();
            reader.onloadend = () => {
              const b64 = (reader.result as string).split(",")[1];
              if (b64)
                fetch(`${BACKEND}/process-frame`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ session_id: sid, frame_base64: b64 }),
                }).catch(() => {});
            };
            reader.readAsDataURL(blob);
          },
          "image/jpeg",
          0.4
        );
      } catch (_) {}
    };

    const onCanPlay = () => sendFrame();
    video.addEventListener("loadeddata", onCanPlay);
    video.addEventListener("playing", onCanPlay);
    frameIntervalRef.current = setInterval(sendFrame, FRAME_INTERVAL_MS);
    return () => {
      video.removeEventListener("loadeddata", onCanPlay);
      video.removeEventListener("playing", onCanPlay);
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    };
  }, [isStarted, sessionId]);

  useEffect(() => {
    if (!isStarted) return;
    timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isStarted]);

  const handleEndCall = async () => {
    if (isEnding) return;
    setIsEnding(true);
    if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    frameIntervalRef.current && clearInterval(frameIntervalRef.current);

    const conv = messages.filter((m) => m.role !== "system").map((m) => ({ role: m.role, content: m.content }));
    const sid = sessionId || sessionIdRef.current || "local-" + Date.now();
    sessionStorage.setItem("feedback_session_id", sid);
    sessionStorage.setItem("feedback_conversation", JSON.stringify(conv));
    router.push("/feedback");
  };

  const handleEndCallRef = useRef(handleEndCall);
  handleEndCallRef.current = handleEndCall;
  useEffect(() => {
    if (elapsed * 1000 >= INTERVIEW_DURATION_MS) handleEndCallRef.current();
  }, [elapsed]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  if (!isStarted) {
    return (
      <main className="min-h-screen bg-navy-900 flex flex-col items-center justify-center p-6">
        <h1 className="text-5xl font-serif text-gold-500 mb-4 tracking-wide">Almost There</h1>
        <p className="text-white/50 mb-8">When you're ready, suit up.</p>
        <button
          onClick={startInterview}
          className="px-12 py-4 bg-gold-500 text-navy-900 font-bold tracking-widest rounded-none hover:bg-gold-400 transition-colors"
        >
          START INTERVIEW
        </button>
      </main>
    );
  }

  return (
    <main className="h-screen bg-navy-900 text-white overflow-hidden flex flex-col">
      <canvas ref={canvasRef} className="hidden" />
      {backendError && (
        <div className="bg-amber-900/50 border-b border-amber-500/50 px-4 py-2 text-amber-200 text-sm text-center">
          {backendError}
        </div>
      )}
      <div className="flex-1 flex min-h-0">
        <div className="flex-1 relative flex items-center justify-center bg-navy-800 p-4 min-w-0">
          {/* Barney fills the entire allocated space */}
          <div className="absolute inset-4 rounded-2xl overflow-hidden border border-gold-500/30 shadow-2xl bg-navy-900">
            <img src="/Barney.jpg" alt="Interviewer" className="absolute inset-0 w-full h-full object-cover object-center" />
            <div className="absolute bottom-4 left-4 bg-black/60 px-3 py-1.5 rounded text-sm text-gold-500 border border-gold-500/30">
              Barney (Hiring Manager)
            </div>
          </div>
          {/* User camera: fixed corner overlay so it's always visible */}
          <div className="absolute bottom-6 right-6 w-48 h-36 rounded-xl border-2 border-gold-500/30 overflow-hidden shadow-xl bg-navy-900 z-10">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover -scale-x-100" />
            <div className="absolute bottom-0 left-0 right-0 py-1 bg-black/60 text-center text-[10px] text-white/80">You</div>
          </div>
          <div className="absolute top-6 right-6 bg-black/60 px-4 py-2 rounded font-mono text-gold-500 z-10">
            {formatTime(elapsed)}
          </div>
        </div>

        <div className="w-[400px] bg-navy-800/80 border-l border-gold-500/20 flex flex-col">
          <div className="p-4 border-b border-gold-500/20 font-bold text-gold-500 tracking-wider">Meeting Chat</div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages
              .filter((m) => m.role !== "system")
              .map((m, i) => (
                <div key={i} className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}>
                  <span className="text-[10px] text-gold-500/60 mb-1 uppercase">{m.role === "user" ? "You" : "Interviewer"}</span>
                  <div
                    className={`p-3 rounded-lg text-sm ${
                      m.isThinking
                        ? "bg-gold-500/20 text-gold-400 animate-pulse border border-gold-500/30"
                        : m.role === "user"
                          ? "bg-gold-500/30 border border-gold-500/40"
                          : "bg-white/5 border border-white/10"
                    }`}
                  >
                    {m.content}
                  </div>
                </div>
              ))}
            <div ref={chatEndRef} />
          </div>
          <div className="p-3 text-center text-xs text-white/50">
            {isTranscribing ? "Processing..." : isRecording ? "Recording — click mic when done" : "Click mic to speak your answer"}
          </div>
        </div>
      </div>

      <div className="flex justify-center py-4 bg-navy-800/50 border-t border-gold-500/20">
        <div className="flex gap-4 items-center">
          <button
            onClick={toggleRecording}
            disabled={isTranscribing || isSpeaking}
            className={`p-4 rounded-full transition-colors disabled:opacity-50 ${
              isRecording ? "bg-red-500" : "bg-gold-500/20 hover:bg-gold-500/40 text-gold-500"
            }`}
          >
            {isTranscribing ? <Loader2 className="w-5 h-5 animate-spin" /> : isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </button>
          <button
            onClick={handleEndCall}
            disabled={isEnding}
            className="flex items-center gap-2 bg-red-600 hover:bg-red-500 px-8 py-3 rounded-lg font-bold text-sm uppercase disabled:opacity-50"
          >
            <PhoneOff className="w-4 h-4" /> End Call
          </button>
        </div>
      </div>
    </main>
  );
}