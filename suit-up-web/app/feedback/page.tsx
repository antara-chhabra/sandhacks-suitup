"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { RotateCcw, LogOut, AlertCircle } from "lucide-react";

const BACKEND = typeof process !== "undefined" && process.env.NEXT_PUBLIC_BACKEND_URL
  ? process.env.NEXT_PUBLIC_BACKEND_URL
  : "http://localhost:8000";

export default function FeedbackPage() {
  const router = useRouter();
  const [report, setReport] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const sid = sessionStorage.getItem("feedback_session_id");
    let conversation: { role: string; content: string }[] = [];
    try {
      const convStr = sessionStorage.getItem("feedback_conversation");
      conversation = convStr ? JSON.parse(convStr) : [];
    } catch {
      conversation = [];
    }

    if (!sid) {
      setError("No session found. Start an interview first.");
      setIsLoading(false);
      return;
    }

    const fetchReport = async () => {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 90000);

        const res = await fetch(`${BACKEND}/end-interview`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sid, conversation }),
          signal: controller.signal,
        });
        clearTimeout(timeout);

        if (!res.ok) {
          const errBody = await res.text();
          let errMsg = "Failed to generate report";
          try {
            const errJson = JSON.parse(errBody);
            errMsg = errJson.detail || errJson.error || errMsg;
          } catch {
            errMsg = errBody || errMsg;
          }
          throw new Error(errMsg);
        }

        const data = await res.json();
        let text = data.feedback?.report_text || data.report_text || "Report generated.";
        // Remove legacy/error lines from display
        text = text
          .split("\n")
          .filter((line: string) => !line.includes("Visual analysis unavailable") && !/^Error:?/i.test(line.trim()))
          .join("\n");
        setReport(text);
        setError(null);
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        const isOffline = msg.includes("fetch") || msg.includes("Network") || msg.includes("Failed to fetch");
        setError(
          isOffline
            ? "Backend offline. Start it with: cd suit-up-web && npm run backend"
            : `Could not generate report: ${msg}`
        );
        setReport(
          "INTERVIEW PERFORMANCE REPORT\n" +
            "============================================================\n\n" +
            "Report generation could not be completed.\n" +
            "Please ensure the Python backend is running: npm run backend (in suit-up-web)\n" +
            "Then retry the interview for a full report.\n\n" +
            "You can still proceed below."
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchReport();
  }, []);

  const handleThankYou = () => {
    sessionStorage.removeItem("feedback_session_id");
    sessionStorage.removeItem("feedback_conversation");
    router.push("/thankyou");
  };

  const handleRetry = () => {
    sessionStorage.removeItem("feedback_session_id");
    sessionStorage.removeItem("feedback_conversation");
    router.push("/company_position");
  };

  return (
    <main className="min-h-screen bg-navy-900 flex flex-col items-center p-6">
      <h1 className="text-4xl font-serif text-gold-500 mb-2 tracking-wide">Your Report</h1>
      <p className="text-white/50 text-sm tracking-widest uppercase mb-8">Interview feedback</p>

      {isLoading ? (
        <div className="flex flex-col items-center justify-center flex-1 w-full max-w-2xl">
          <div className="relative w-full max-w-md aspect-video rounded-2xl overflow-hidden border border-gold-500/30 mb-8">
            <img src="/Barney.jpg" alt="Barney" className="w-full h-full object-cover animate-pulse" />
            <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
              <p className="text-gold-500 text-xl font-serif animate-pulse">Suit up! Generating your legendary report...</p>
            </div>
          </div>
          <p className="text-white/40 text-sm">Analyzing your performance...</p>
        </div>
      ) : (
        <>
          {error && (
            <div className="w-full max-w-2xl mb-4 flex items-center gap-2 p-4 bg-amber-900/30 border border-amber-500/50 rounded-lg text-amber-200 text-sm">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              {error}
            </div>
          )}
          <div className="w-full max-w-2xl bg-navy-800/50 border border-gold-500/20 rounded-xl overflow-hidden">
            <div className="max-h-[60vh] overflow-y-auto p-6 font-mono text-sm text-white/90 whitespace-pre-wrap leading-relaxed scroll-smooth">
              {report || "No report available."}
            </div>
            <p className="text-center text-white/40 text-xs py-2 border-t border-gold-500/10">Scroll for full report</p>
          </div>

          <div className="flex gap-6 mt-12">
            <button
              onClick={handleThankYou}
              className="flex items-center gap-2 px-8 py-4 bg-gold-500 text-navy-900 font-bold tracking-widest rounded-none hover:bg-gold-400 transition-colors"
            >
              <LogOut className="w-5 h-5" /> Thank You & Exit
            </button>
            <button
              onClick={handleRetry}
              className="flex items-center gap-2 px-8 py-4 bg-transparent border border-gold-500/50 text-gold-500 font-bold tracking-widest rounded-none hover:bg-gold-500 hover:text-navy-900 transition-colors"
            >
              <RotateCcw className="w-5 h-5" /> Retry Interview
            </button>
          </div>
        </>
      )}
    </main>
  );
}
