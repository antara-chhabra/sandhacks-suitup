"use client";
import React, { useState, useEffect } from "react";
import { Building2, Briefcase, ChevronRight, Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";

export default function CompanyPositionPage() {
  const router = useRouter();
  const [company, setCompany] = useState("");
  const [position, setPosition] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!localStorage.getItem("interview_questions")) {
      router.replace("/upload_resume");
    }
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!company.trim() || !position.trim()) {
      alert("Please enter both company and position.");
      return;
    }
    setIsLoading(true);

    try {
      const BACKEND = typeof process !== "undefined" && process.env.NEXT_PUBLIC_BACKEND_URL
        ? process.env.NEXT_PUBLIC_BACKEND_URL
        : "http://localhost:8000";
      const res = await fetch(`${BACKEND}/generate-company-questions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ company: company.trim(), position: position.trim() }),
      });
      if (!res.ok) throw new Error("Failed to generate questions");
      const data = await res.json();
      localStorage.setItem("company_questions", JSON.stringify(data));
      localStorage.setItem("company", company.trim());
      localStorage.setItem("position", position.trim());
      router.push("/interview_setup");
    } catch (err) {
      console.error(err);
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("fetch") || msg.includes("Network")) {
        alert("Backend offline. Run: cd suit-up-web && npm run backend");
      }
      // Fallback: save locally and proceed
      localStorage.setItem("company_questions", JSON.stringify({
        company: company.trim(),
        position: position.trim(),
        company_values: [],
        technical_questions: [],
        behavioral_questions: [],
      }));
      localStorage.setItem("company", company.trim());
      localStorage.setItem("position", position.trim());
      router.push("/interview_setup");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-navy-900 flex items-center justify-center p-6">
      <div className="w-full max-w-lg flex flex-col items-center">
        <h1 className="text-5xl md:text-6xl font-serif text-gold-500 mb-2 tracking-wide">
          Where to?
        </h1>
        <p className="text-white/50 text-lg font-light tracking-wider mb-12">
          Company and role for tailored interview questions
        </p>

        <div className="w-full flex items-center justify-center my-8">
          <div className="h-[1px] flex-grow bg-gradient-to-r from-transparent to-gold-500/50" />
          <div className="h-[2px] w-48 bg-gold-500 shadow-[0_0_20px_rgba(197,160,89,0.4)" />
          <div className="h-[1px] flex-grow bg-gradient-to-l from-transparent to-gold-500/50" />
        </div>

        <form onSubmit={handleSubmit} className="w-full max-w-sm space-y-6">
          <div className="relative group">
            <Building2 className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gold-500/40 group-focus-within:text-gold-500 transition-colors" />
            <input
              type="text"
              value={company}
              onChange={(e) => setCompany(e.target.value)}
              className="block w-full pl-12 pr-4 py-4 bg-white/5 border border-white/10 rounded-none text-white placeholder:text-white/20 focus:outline-none focus:border-gold-500/50 focus:bg-white/10 transition-all font-light tracking-widest uppercase text-sm"
              placeholder="Company"
            />
          </div>

          <div className="relative group">
            <Briefcase className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gold-500/40 group-focus-within:text-gold-500 transition-colors" />
            <input
              type="text"
              value={position}
              onChange={(e) => setPosition(e.target.value)}
              className="block w-full pl-12 pr-4 py-4 bg-white/5 border border-white/10 rounded-none text-white placeholder:text-white/20 focus:outline-none focus:border-gold-500/50 focus:bg-white/10 transition-all font-light tracking-widest uppercase text-sm"
              placeholder="Position / Role"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full mt-4 bg-transparent border border-gold-500/50 text-gold-500 py-4 rounded-none font-bold tracking-[0.3em] text-xs hover:bg-gold-500 hover:text-navy-900 transition-all duration-700 flex items-center justify-center gap-2 group cursor-pointer disabled:opacity-50"
          >
            {isLoading ? (
              <>Generating questions... <Loader2 className="w-4 h-4 animate-spin" /></>
            ) : (
              <>Continue <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" /></>
            )}
          </button>
        </form>

        <div className="mt-16 opacity-20">
          <p className="text-white text-[10px] tracking-[0.5em] uppercase font-light">TEAM SAAS</p>
        </div>
      </div>
    </main>
  );
}
