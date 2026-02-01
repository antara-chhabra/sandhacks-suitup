"use client";
import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/login');
  }, [router]);

  return (
    <main className="min-h-screen bg-navy-900 flex items-center justify-center p-6">
      <div className="w-full max-w-lg flex flex-col items-center">
        
        <h1 className="text-7xl font-light tracking-[0.2em] text-gold-500 mb-2 font-serif">
          SuitUp
        </h1>
        <p className="text-white/40 text-lg italic tracking-wider font-light">
          For your Legen—wait for it—dary interviewing.
        </p>
        <div className="mt-12">
          <p className="text-gold-500/60 animate-pulse">Redirecting to login...</p>
        </div>
      </div>
    </main>
  );
}
