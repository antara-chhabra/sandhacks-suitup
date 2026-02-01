"use client";
import { useState, useEffect, useRef } from 'react';

declare global {
  interface Window { webkitSpeechRecognition: any; }
}

export default function MockInterview() {
  const [role, setRole] = useState("");
  const [company, setCompany] = useState("");
  const [messages, setMessages] = useState<any[]>([]);
  const [isStarted, setIsStarted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [jawOffset, setJawOffset] = useState(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const recognitionRef = useRef<any>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  // Setup Webcam Feed
  useEffect(() => {
    if (isStarted && videoRef.current) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((s) => { if (videoRef.current) videoRef.current.srcObject = s; });
    }
  }, [isStarted]);

  // Voice Recognition with Auto-Send
  useEffect(() => {
    if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.onresult = (e: any) => handleReply(e.results[0][0].transcript);
      recognition.onend = () => setIsListening(false);
      recognitionRef.current = recognition;
    }
  }, [messages]);

  const startListening = () => {
    if (isSpeaking) return;
    setIsListening(true);
    recognitionRef.current?.start();
  };

  // Synchronized Speech and Typewriter logic
  const speakWithSync = (text: string) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    utterance.voice = voices.find(v => v.name.includes('Male') || v.lang.includes('en-GB')) || voices[0];
    utterance.rate = 0.95;

    // Create slot for AI response
    setMessages(prev => [...prev, { role: 'assistant', content: "" }]);

    // SYNC: Update text only as words are spoken
    utterance.onboundary = (event) => {
      if (event.name === 'word') {
        const spokenPart = text.slice(0, event.charIndex + (text.slice(event.charIndex).split(' ')[0].length));
        setMessages(prev => {
          const last = prev[prev.length - 1];
          return [...prev.slice(0, -1), { ...last, content: spokenPart }];
        });
      }
    };

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => {
      setIsSpeaking(false);
      setMessages(prev => {
        const last = prev[prev.length - 1];
        return [...prev.slice(0, -1), { ...last, content: text }];
      });
      setTimeout(startListening, 600);
    };
    window.speechSynthesis.speak(utterance);
  };

  const getAIResponse = async (history: any[]) => {
    // Show "Thinking..."
    setMessages(prev => [...prev, { role: 'assistant', content: 'Thinking...', isThinking: true }]);
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history }),
      });
      const data = await res.json();
      const aiText = data.message?.content || "I'm sorry, I'm having trouble connecting.";
      
      // Remove Thinking and Start Sync Output
      setMessages(prev => prev.filter(m => !m.isThinking));
      speakWithSync(aiText);
    } catch (e) {
      setMessages(prev => prev.filter(m => !m.isThinking));
    }
  };

  const handleReply = async (text: string) => {
    if (!text.trim()) return;
    const newHistory = [...messages, { role: 'user', content: text }];
    setMessages(newHistory);
    await getAIResponse(newHistory);
  };

  const startInterview = async () => {
    const system = { role: 'system', content: `You are an interviewer at ${company} for ${role}. One question at a time.` };
    setMessages([system]);
    setIsStarted(true);
    await getAIResponse([system, { role: 'user', content: 'Start.' }]);
  };

  // Jaw Sync
  useEffect(() => {
    let interval: any;
    if (isSpeaking) interval = setInterval(() => setJawOffset(Math.random() * 12), 80);
    else setJawOffset(0);
    return () => clearInterval(interval);
  }, [isSpeaking]);

  return (
    <main className="h-screen bg-[#121212] text-white overflow-hidden font-sans">
      {!isStarted ? (
        <div className="flex flex-col items-center justify-center h-full gap-6">
          <h1 className="text-7xl font-black italic text-blue-500 tracking-tighter">SUIT UP</h1>
          <div className="bg-zinc-900 p-8 rounded-3xl border border-zinc-800 flex flex-col gap-4 w-96 shadow-2xl">
            <input className="bg-black p-4 rounded-xl border border-zinc-800" placeholder="Company" onChange={e => setCompany(e.target.value)} />
            <input className="bg-black p-4 rounded-xl border border-zinc-800" placeholder="Role" onChange={e => setRole(e.target.value)} />
            <button onClick={startInterview} className="bg-white text-black py-4 rounded-xl font-bold uppercase tracking-widest">Join Interview</button>
          </div>
        </div>
      ) : (
        <div className="flex h-full w-full">
          {/* Main Stage (Interviewer) */}
          <div className="flex-1 relative flex items-center justify-center bg-black p-12">
            <div className="relative w-full max-w-4xl aspect-video rounded-3xl overflow-hidden border border-zinc-800 shadow-2xl bg-zinc-900">
              <img src="/Barney.jpg" className="w-full h-full object-cover" />
              <div className="absolute inset-0 w-full h-full" style={{ clipPath: 'inset(60% 20% 10% 20%)', transform: `translateY(${jawOffset}px)` }}>
                <img src="/Barney.jpg" className="w-full h-full object-cover" />
              </div>
              <div className="absolute bottom-6 left-6 bg-black/60 px-4 py-2 rounded-lg text-sm">Barney (Hiring Manager)</div>
            </div>

            {/* Candidate Webcam */}
            <div className="absolute bottom-10 left-10 w-64 h-40 bg-zinc-900 rounded-2xl border-2 border-zinc-700 overflow-hidden shadow-2xl">
              <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover -scale-x-100" />
            </div>
          </div>

          {/* Chat Panel */}
          <div className="w-[420px] bg-zinc-900 border-l border-zinc-800 flex flex-col">
            <div className="p-6 border-b border-zinc-800 font-bold tracking-tight">Meeting Chat</div>
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {messages.filter(m => m.role !== 'system').map((m, i) => (
                <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <span className="text-[10px] text-zinc-500 mb-1 uppercase tracking-tighter">{m.role === 'user' ? 'You' : 'Barney'}</span>
                  <div className={`p-4 rounded-2xl text-sm leading-relaxed ${
                    m.isThinking ? 'bg-zinc-800 text-blue-400 animate-pulse border border-blue-900' :
                    m.role === 'user' ? 'bg-blue-600' : 'bg-zinc-800 border border-zinc-700'
                  }`}>
                    {m.content}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
            <div className={`p-4 text-center text-[10px] font-mono ${isListening ? 'text-red-500' : 'text-zinc-600'}`}>
              {isListening ? "● RECORDING AUDIO" : "AWAITING RESPONSE"}
            </div>
          </div>

          {/* Controls */}
          <div className="absolute bottom-8 left-0 right-0 flex justify-center pointer-events-none">
            <div className="flex gap-4 bg-zinc-900/90 backdrop-blur-xl px-8 py-4 rounded-full border border-white/10 pointer-events-auto shadow-2xl">
              <button onClick={startListening} className={`p-4 rounded-full ${isListening ? 'bg-red-500' : 'bg-zinc-800'}`}>🎤</button>
              <button className="p-4 rounded-full bg-zinc-800">📷</button>
              <button onClick={() => window.location.reload()} className="bg-red-600 px-8 rounded-full font-bold text-xs uppercase">End</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}