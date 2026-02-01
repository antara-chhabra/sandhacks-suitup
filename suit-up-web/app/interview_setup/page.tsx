"use client";
import React, { useEffect, useRef, useState } from 'react';
import { Camera, Mic, CheckCircle, AlertCircle, ArrowRight, Loader2 } from 'lucide-react';
import Link from 'next/link';

export default function InterviewSetupPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const [hasMicPermission, setHasMicPermission] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState(true);
  
  useEffect(() => {
    let stream: MediaStream | null = null;

    const startCamera = async () => {
      try {
        // Request permissions
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 1280, height: 720 }, // Requesting specific resolution helps sometimes
          audio: true 
        });
        
        // Success state updates
        setHasCameraPermission(true);
        setHasMicPermission(true);

        // Assign stream to video element
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

      } catch (err) {
        console.error("Error accessing media devices:", err);
        setHasCameraPermission(false);
        setHasMicPermission(false);
      } finally {
        // Stop the "Initializing..." spinner after a short delay
        setTimeout(() => setIsChecking(false), 1000);
      }
    };

    startCamera();

    // Cleanup function: Only stop tracks when the component actually unmounts (leaving the page)
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <main className="min-h-screen bg-navy-900 flex flex-col items-center justify-center p-6 text-white">
      
      {/* Title */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-serif text-gold-500 mb-2">The Green Room</h1>
        <p className="text-white/50 text-sm tracking-widest uppercase">Check your fit before you commit</p>
      </div>

      <div className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
        
        {/* LEFT SIDE: The Mirror (Video Feed) */}
        <div className="relative group">
          {/* Gold Border Frame */}
          <div className="absolute -inset-1 bg-gradient-to-br from-gold-500/20 to-navy-900 rounded-2xl blur-sm transition-all duration-500 group-hover:from-gold-500/40"></div>
          
          <div className="relative bg-navy-800 rounded-2xl overflow-hidden border border-white/10 shadow-2xl aspect-video flex items-center justify-center">
            
            {/* 1. INITIALIZING STATE */}
            {isChecking && (
              <div className="absolute inset-0 z-20 bg-navy-800 flex flex-col items-center justify-center gap-3">
                <Loader2 className="w-8 h-8 text-gold-500 animate-spin" />
                <span className="text-xs tracking-widest text-gold-500">INITIALIZING SYSTEMS...</span>
              </div>
            )}

            {/* 2. SUCCESS STATE (Camera Feed) */}
            {hasCameraPermission !== false && (
              <>
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted 
                  // Force play when data loads
                  onLoadedMetadata={() => videoRef.current?.play()} 
                  className={`w-full h-full object-cover transform -scale-x-100 transition-opacity duration-500 ${isChecking ? 'opacity-0' : 'opacity-100'}`}
                />
                
                {/* Overlay UI */}
                {!isChecking && (
                  <>
                    <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1 bg-black/40 backdrop-blur-md rounded-full border border-white/10 z-10">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="text-[10px] font-bold tracking-wider text-white/80">LIVE</span>
                    </div>

                    {/* Scanning Line Animation */}
                    <div className="absolute inset-0 pointer-events-none overflow-hidden z-10">
                      <div className="w-full h-[1px] bg-gold-500/50 shadow-[0_0_15px_rgba(197,160,89,0.8)] animate-[scan_3s_ease-in-out_infinite] opacity-30"></div>
                    </div>
                  </>
                )}
              </>
            )}

            {/* 3. ERROR STATE */}
            {!isChecking && hasCameraPermission === false && (
              <div className="text-center p-8 z-20">
                <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
                <h3 className="text-xl font-bold mb-2">Camera Access Denied</h3>
                <p className="text-white/50 text-sm">We need to see that beautiful face to analyze your interview skills.</p>
                <button 
                  onClick={() => window.location.reload()}
                  className="mt-6 px-6 py-2 bg-white/10 hover:bg-white/20 rounded-full text-sm transition"
                >
                  Try Again
                </button>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT SIDE: System Checks */}
        <div className="space-y-6">
          
          <div className="bg-navy-800/50 border border-white/5 rounded-xl p-6 backdrop-blur-sm">
            <h3 className="text-gold-500 text-xs font-bold tracking-[0.2em] mb-6 uppercase">System Diagnostics</h3>
            
            {/* Camera Check Item */}
            <div className="flex items-center justify-between mb-6 group">
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-full ${hasCameraPermission ? 'bg-green-500/20 text-green-400' : 'bg-white/5 text-white/40'}`}>
                  <Camera className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="font-medium text-white">Video Feed</h4>
                  <p className="text-xs text-white/40">{hasCameraPermission ? "Operational" : "Not detected"}</p>
                </div>
              </div>
              {hasCameraPermission && <CheckCircle className="w-5 h-5 text-green-500" />}
            </div>

            {/* Mic Check Item */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-full ${hasMicPermission ? 'bg-green-500/20 text-green-400' : 'bg-white/5 text-white/40'}`}>
                  <Mic className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="font-medium text-white">Audio Input</h4>
                  <p className="text-xs text-white/40">{hasMicPermission ? "Microphone active" : "Check permissions"}</p>
                </div>
              </div>
              {hasMicPermission && <CheckCircle className="w-5 h-5 text-green-500" />}
            </div>
          </div>

          {/* Action Button */}
          <div className="pt-4">
            <Link 
              href="/interview_session" 
              className={`
                group w-full flex items-center justify-center gap-3 py-4 
                bg-gold-500 hover:bg-gold-400 text-navy-900 
                font-bold tracking-widest text-sm rounded-lg transition-all
                ${(hasCameraPermission && hasMicPermission) ? 'opacity-100 transform translate-y-0' : 'opacity-50 cursor-not-allowed pointer-events-none'}
              `}
            >
              START SESSION <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
            <p className="text-center text-white/20 text-[10px] mt-4 uppercase tracking-widest">
              By clicking start, you agree to be legendary.
            </p>
          </div>

        </div>
      </div>
    </main>
  );
}
