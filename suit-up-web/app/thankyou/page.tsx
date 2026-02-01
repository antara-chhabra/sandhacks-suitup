"use client";

import { useRouter } from "next/navigation";
import { Heart } from "lucide-react";

export default function ThankYouPage() {
  const router = useRouter();

  return (
    <main className="min-h-screen bg-navy-900 flex flex-col items-center justify-center p-6">
      <h1 className="text-6xl font-serif text-gold-500 mb-4 tracking-wide">
        Thank You
      </h1>
      <p className="text-white/60 text-lg font-light tracking-wider mb-2">
        You were legen—wait for it—dary!
      </p>
      <p className="text-white/40 text-sm mb-12">
        We hope SuitUp helped you feel more confident. Good luck with your interviews.
      </p>

      <div className="w-full flex items-center justify-center my-8">
        <div className="h-[1px] flex-grow bg-gradient-to-r from-transparent to-gold-500/50" />
        <div className="h-[2px] w-48 bg-gold-500 shadow-[0_0_20px_rgba(197,160,89,0.4)" />
        <div className="h-[1px] flex-grow bg-gradient-to-l from-transparent to-gold-500/50" />
      </div>

      <div className="flex gap-4">
        <button
          onClick={() => router.push("/login")}
          className="flex items-center gap-2 px-8 py-4 bg-transparent border border-gold-500/50 text-gold-500 font-bold tracking-widest rounded-none hover:bg-gold-500 hover:text-navy-900 transition-colors"
        >
          Start Again
        </button>
      </div>

      <div className="mt-16 opacity-20">
        <p className="text-white text-[10px] tracking-[0.5em] uppercase font-light flex items-center gap-2">
          <Heart className="w-3 h-3" /> TEAM SAAS
        </p>
      </div>
    </main>
  );
}
