"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Mic, MicOff, PhoneOff, Loader2 } from "lucide-react";

const BACKEND = "http://localhost:8000";
const INTERVIEW_DURATION_MS = 40 * 60 * 1000;
const FRAME_INTERVAL_MS = 5000; // Reduce from 2s to 5s to avoid camera delay
const AUDIO_CHUNK_MS = 15000;
const SILENCE_THRESHOLD_MS = 5000; // 5 seconds of no speech before AI responds

export default function InterviewSessionPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<{ role: string; content: string; isThinking?: boolean }[]>([]);
  const messagesRef = useRef<{ role: string; content: string; isThinking?: boolean }[]>([]);
  messagesRef.current = messages;
  const [isStarted, setIsStarted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [jawOffset, setJawOffset] = useState(0);
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
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const handleReplyRef = useRef<((text: string) => Promise<void>) | null>(null);
  const volumeIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
    return `You are Barney Stinson from How I Met Your Mother, acting as a charismatic interviewer at ${company} for the ${position} role. Stay in character - confident, smooth, occasional catchphrases. Ask ONE question at a time. Mix these question themes naturally: ${allQuestions.slice(0, 8).join(" | ")}. Keep responses concise. After the candidate answers, ask the next question or wrap up after 5-7 exchanges.`;
  }, []);

  const speakWithSync = useCallback((text: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    utterance.voice = voices.find((v) => v.name.includes("Male") || v.lang.includes("en-GB")) || voices[0];
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
      setIsRecording(true); // Resume listening after AI speaks
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

  const stopRecordingAndTranscribe = useCallback(async () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    setIsRecording(false);
    setIsTranscribing(true);

    recorder.stop();
    await new Promise<void>((resolve) => {
      const onStop = () => {
        recorder.removeEventListener("stop", onStop);
        resolve();
      };
      recorder.addEventListener("stop", onStop);
    });

    const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
    audioChunksRef.current = [];

    if (blob.size < 1000) {
      setIsTranscribing(false);
      setIsRecording(true);
      return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
      const b64 = (reader.result as string).split(",")[1];
      if (!b64) {
        setIsTranscribing(false);
        setIsRecording(true);
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
        const transcript = data.transcription?.trim() || "";
        if (transcript && handleReplyRef.current) {
          await handleReplyRef.current(transcript);
        }
      } catch (e) {
        console.error("Transcribe error:", e);
        setBackendError("Could not transcribe. Is the backend running on port 8000?");
      } finally {
        setIsTranscribing(false);
        setIsRecording(true);
        mediaRecorderRef.current?.start(AUDIO_CHUNK_MS);
      }
    };
    reader.readAsDataURL(blob);
  }, []);

  const toggleRecording = useCallback(() => {
    if (isTranscribing || isSpeaking) return;
    if (isRecording) {
      stopRecordingAndTranscribe();
    } else {
      setIsRecording(true);
      mediaRecorderRef.current?.start(AUDIO_CHUNK_MS);
    }
  }, [isRecording, isTranscribing, isSpeaking, stopRecordingAndTranscribe]);

  const startInterview = useCallback(async () => {
    setBackendError(null);
    const resumeQuestions = JSON.parse(localStorage.getItem("interview_questions") || "[]");
    const companyQuestions = JSON.parse(localStorage.getItem("company_questions") || "{}");
    const company = localStorage.getItem("company") || "Company";
    const position = localStorage.getItem("position") || "Role";

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
    await getAIResponse([system, { role: "user", content: "Start the interview." }]);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: 15 },
        audio: true,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        audioChunksRef.current = [];
        const sid = sessionIdRef.current;
        if (sid && blob.size > 1000) {
          const r = new FileReader();
          r.onloadend = () => {
            const b64 = (r.result as string).split(",")[1];
            if (b64)
              fetch(`${BACKEND}/process-audio`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sid, audio_base64: b64 }),
              }).catch(() => {});
          };
          r.readAsDataURL(blob);
        }
      };

      recorder.start(AUDIO_CHUNK_MS);
      setIsRecording(true);

      // Silence detection: after 5 sec of low volume, auto-submit
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      let lastLoudTime = Date.now();

      const checkVolume = () => {
        if (isTranscribing || isSpeaking) return;
        if (!recorder || recorder.state !== "recording") return;
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        if (avg > 15) lastLoudTime = Date.now();
        else if (Date.now() - lastLoudTime > SILENCE_THRESHOLD_MS && audioChunksRef.current.length > 0) {
          lastLoudTime = Date.now();
          stopRecordingAndTranscribe();
        }
      };
      volumeIntervalRef.current = setInterval(checkVolume, 500);
    } catch (err) {
      console.error("Media error:", err);
      setBackendError("Could not access camera/microphone. Check permissions.");
    }
  }, [buildSystemPrompt, getAIResponse, stopRecordingAndTranscribe]);

  // Frame capture - reduced frequency, lower quality
  useEffect(() => {
    if (!isStarted || !sessionId || !videoRef.current) return;
    const sendFrame = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      canvas.width = 160;
      canvas.height = 120;
      ctx.drawImage(video, 0, 0, 160, 120);
      canvas.toBlob(
        (blob) => {
          if (!blob) return;
          const reader = new FileReader();
          reader.onloadend = () => {
            const b64 = (reader.result as string).split(",")[1];
            if (b64)
              fetch(`${BACKEND}/process-frame`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, frame_base64: b64 }),
              }).catch(() => {});
          };
          reader.readAsDataURL(blob);
        },
        "image/jpeg",
        0.3
      );
    };
    frameIntervalRef.current = setInterval(sendFrame, FRAME_INTERVAL_MS);
    return () => {
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
    volumeIntervalRef.current && clearInterval(volumeIntervalRef.current);
    mediaRecorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    frameIntervalRef.current && clearInterval(frameIntervalRef.current);
    audioContextRef.current?.close();

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

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isSpeaking) interval = setInterval(() => setJawOffset(Math.random() * 12), 80);
    else setJawOffset(0);
    return () => clearInterval(interval);
  }, [isSpeaking]);

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
        <div className="flex-1 relative flex items-center justify-center bg-navy-800 p-6">
          <div className="relative w-full max-w-4xl aspect-video rounded-2xl overflow-hidden border border-gold-500/30 shadow-2xl bg-navy-900">
            <img src="/Barney.jpg" alt="Interviewer" className="w-full h-full object-cover" />
            <div
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ clipPath: "inset(60% 20% 10% 20%)", transform: `translateY(${jawOffset}px)` }}
            >
              <img src="/Barney.jpg" alt="" className="w-full h-full object-cover" />
            </div>
            <div className="absolute bottom-4 left-4 bg-black/60 px-3 py-1.5 rounded text-sm text-gold-500 border border-gold-500/30">
              Barney (Hiring Manager)
            </div>
          </div>
          <div className="absolute bottom-6 left-6 w-56 h-36 bg-navy-900 rounded-xl border-2 border-gold-500/30 overflow-hidden shadow-xl">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover -scale-x-100" />
          </div>
          <div className="absolute top-6 right-6 bg-black/60 px-4 py-2 rounded font-mono text-gold-500">
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
                  <span className="text-[10px] text-gold-500/60 mb-1 uppercase">{m.role === "user" ? "You" : "Barney"}</span>
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
          <div className="p-3 text-center text-xs font-mono text-white/40">
            {isTranscribing ? "Transcribing with Whisper..." : isRecording ? "● Recording... Click mic when done (or wait 5s silence)" : "Click mic to start recording your answer"}
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
