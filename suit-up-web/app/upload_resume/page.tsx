"use client";
import React, { useState, useRef } from "react";
import { Upload, CheckCircle, ArrowRight, Loader2 } from "lucide-react"; // Added Loader2 for the 'Analyzing' state
import { useRouter } from "next/navigation"; // Added for manual navigation

export default function UploadPage() {
  const router = useRouter(); // Initialize router
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false); // New state to track backend processing
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (uploadedFile: File) => {
    if (uploadedFile.type === "application/pdf") {
      setFile(uploadedFile);
    } else {
      alert("Please upload a PDF file.");
    }
  };

  /**
   * NEW LOGIC: This sends the PDF to your Python backend (Llama)
   * and saves the questions before moving to the next page.
   */
  const handleChallengeAccepted = async () => {
    if (!file) return;

    setIsUploading(true);
    
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Connects to your FastAPI backend on port 8000
        const response = await fetch("http://localhost:8000/process-resume", {
  method: "POST",
  body: formData,
      });

      if (!response.ok) throw new Error("Processing failed");

      const data = await response.json();

      // Store the Llama-generated questions in localStorage for the interview session
      localStorage.setItem("interview_questions", JSON.stringify(data.questions));
      
      // Proceed to company/position page
      router.push('/company_position');
    } catch (error) {
      console.error("AI Error:", error);
      const msg = error instanceof Error ? error.message : String(error);
      alert(
        msg.includes("fetch") || msg.includes("Network")
          ? "Backend offline. Run: cd suit-up-web && npm run backend"
          : "Resume processing failed. Is the backend running on port 8000?"
      );
    } finally {
      setIsUploading(false);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <main className="min-h-screen bg-navy-900 flex flex-col items-center justify-center p-6 relative overflow-hidden">
      
      {/* Background Decoration */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-gold-500/30 to-transparent"></div>

      <div className="w-full max-w-2xl flex flex-col items-center z-10">
        
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in-up">
          <h1 className="text-5xl md:text-6xl font-serif text-gold-500 mb-4 tracking-wide">
            Welcome in.
          </h1>
          <p className="text-white/60 text-lg font-light tracking-wider">
            A lie is just a great story that someone ruined with the truth.<br />
            Let's see your resume.
          </p>
        </div>

        {/* Upload Card */}
        <div 
          onClick={triggerFileInput}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            w-full bg-navy-800/50 backdrop-blur-sm border-2 border-dashed rounded-xl p-12
            flex flex-col items-center justify-center cursor-pointer transition-all duration-300
            group hover:bg-navy-800/80
            ${isDragging ? "border-gold-500 bg-navy-800/80 scale-[1.02]" : "border-gold-500/30 hover:border-gold-500/60"}
          `}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileSelect} 
            accept=".pdf" 
            className="hidden" 
          />

          {file ? (
            <div className="flex flex-col items-center text-center animate-pulse-once">
              <CheckCircle className="w-16 h-16 text-green-500 mb-4" />
              <p className="text-xl text-white font-medium mb-2">{file.name}</p>
              <p className="text-white/40 text-sm uppercase tracking-widest">Awesome!</p>
              <button 
                onClick={(e) => { e.stopPropagation(); setFile(null); }}
                className="mt-6 text-red-400 hover:text-red-300 text-sm underline z-20"
              >
                Remove
              </button>
            </div>
          ) : (
            <>
              <div className="w-20 h-20 rounded-full bg-gold-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <Upload className="w-8 h-8 text-gold-500" />
              </div>
              <h3 className="text-2xl text-white font-light mb-2">Upload your Resume</h3>
              <p className="text-white/40 text-sm tracking-widest uppercase mb-6">PDF FORMAT ONLY</p>
              <div className="px-6 py-2 border border-white/20 rounded-full text-white/60 text-xs hover:bg-white/5 transition-colors">
                Select File or Drag & Drop
              </div>
            </>
          )}
        </div>

        {/* Action Button: Changed from Link to Button to trigger backend logic */}
        <div className={`mt-10 transition-all duration-500 ${file ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'}`}>
            <button 
                onClick={handleChallengeAccepted}
                disabled={isUploading}
                className="group relative inline-flex items-center justify-center px-8 py-4 overflow-hidden font-medium tracking-tighter text-white bg-transparent border border-gold-500 rounded-none cursor-pointer disabled:opacity-50"
            >
              <span className="absolute w-0 h-0 transition-all duration-500 ease-out bg-gold-500 rounded-full group-hover:w-80 group-hover:h-56"></span>
              <span className="relative flex items-center gap-3 font-bold tracking-[0.2em] uppercase text-sm group-hover:text-navy-900 transition-colors">
                  {isUploading ? (
                    <>Analyzing Systems... <Loader2 className="w-4 h-4 animate-spin" /></>
                  ) : (
                    <>Challenge Accepted <ArrowRight className="w-4 h-4" /></>
                  )}
              </span>
            </button>
        </div>

      </div>
    </main>
  );
}