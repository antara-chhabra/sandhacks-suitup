"use client";
import React, { useState } from 'react';
import { User, Lock, ChevronRight } from 'lucide-react';
import { useRouter } from 'next/navigation'; // Import the router

export default function Home() {
  const router = useRouter(); 
  const [isLoading, setIsLoading] = useState(false);

  const handleEnter = (e: React.FormEvent) => {
    e.preventDefault(); // Stop the form from reloading the page
    setIsLoading(true);

    // Fake loading delay for effect, then navigate
    setTimeout(() => {
      // IMPORTANT: Matches your folder name "upload_resume"
      router.push('/upload_resume'); 
    }, 800);
  };

  return (
    <main className="min-h-screen bg-navy-900 flex items-center justify-center p-6">
      <div className="w-full max-w-lg flex flex-col items-center">
        
        <h1 className="text-7xl font-light tracking-[0.2em] text-gold-500 mb-2 font-serif">
          SuitUp
        </h1>
        <p className="text-white/40 text-lg italic tracking-wider font-light">
          For your Legen—wait for it—dary interviewing.
        </p>

        {/* The Golden Band Line */}
        <div className="w-full flex items-center justify-center my-12">
          <div className="h-[1px] flex-grow bg-gradient-to-r from-transparent to-gold-500/50"></div>
          <div className="h-[2px] w-48 bg-gold-500 shadow-[0_0_20px_rgba(197,160,89,0.4)]"></div>
          <div className="h-[1px] flex-grow bg-gradient-to-l from-transparent to-gold-500/50"></div>
        </div>

        {/* Form with onSubmit handler */}
        <form onSubmit={handleEnter} className="w-full max-w-sm space-y-6">
          <div className="relative group">
            <User className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gold-500/40 group-focus-within:text-gold-500 transition-colors" />
            <input
              type="text"
              className="block w-full pl-12 pr-4 py-4 bg-white/5 border border-white/10 rounded-none text-white placeholder:text-white/20 focus:outline-none focus:border-gold-500/50 focus:bg-white/10 transition-all font-light tracking-widest uppercase text-sm"
              placeholder="Username"
            />
          </div>

          <div className="relative group">
            <Lock className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gold-500/40 group-focus-within:text-gold-500 transition-colors" />
            <input
              type="password"
              className="block w-full pl-12 pr-4 py-4 bg-white/5 border border-white/10 rounded-none text-white placeholder:text-white/20 focus:outline-none focus:border-gold-500/50 focus:bg-white/10 transition-all font-light tracking-widest uppercase text-sm"
              placeholder="Password"
            />
          </div>

          <button 
            type="submit"
            className="w-full mt-4 bg-transparent border border-gold-500/50 text-gold-500 py-4 rounded-none font-bold tracking-[0.3em] text-xs hover:bg-gold-500 hover:text-navy-900 transition-all duration-700 flex items-center justify-center gap-2 group cursor-pointer"
          >
            {isLoading ? "LOADING..." : "ENTER"} 
            {!isLoading && <ChevronRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />}
          </button>
        </form>

        <div className="mt-16 opacity-20">
          <p className="text-white text-[10px] tracking-[0.5em] uppercase font-light">TEAM SAAS</p>
        </div>

      </div>
    </main>
  );
}
