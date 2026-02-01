// app/login/page.tsx
"use client";
import { useState } from 'react';
import { Mail, Lock, ChevronRight } from 'lucide-react';
import Link from 'next/link';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <div className="min-h-screen bg-brand-soft flex flex-col">
      {/* Small header for the login page */}
      <div className="p-6">
        <Link href="/" className="font-bold text-2xl text-brand-dark">SuitUp</Link>
      </div>

      <div className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-[450px] bg-white rounded-3xl shadow-2xl shadow-slate-200/50 p-10 border border-gray-100">
          <div className="mb-10 text-center">
            <h2 className="text-3xl font-bold text-brand-dark">Welcome back</h2>
            <p className="text-gray-500 mt-2">Sign in to continue your practice</p>
          </div>

          <form className="space-y-6">
            <div>
              <label className="block text-sm font-semibold text-brand-dark mb-2">Business Email</label>
              <div className="relative">
                <Mail className="absolute left-4 top-3.5 w-5 h-5 text-gray-400" />
                <input 
                  type="email" 
                  placeholder="name@company.com"
                  className="w-full pl-12 pr-4 py-3.5 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-brand-primary outline-none transition"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold text-brand-dark mb-2">Password</label>
              <div className="relative">
                <Lock className="absolute left-4 top-3.5 w-5 h-5 text-gray-400" />
                <input 
                  type="password" 
                  placeholder="••••••••"
                  className="w-full pl-12 pr-4 py-3.5 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-brand-primary outline-none transition"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </div>

            <button className="w-full bg-brand-primary text-white py-4 rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-blue-700 transition shadow-lg shadow-blue-100">
              Sign In <ChevronRight className="w-5 h-5" />
            </button>
          </form>

          <div className="mt-8 text-center text-sm text-gray-500">
            Don't have an account? <Link href="#" className="text-brand-primary font-bold">Create one for free</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
