// app/layout.tsx
import { Inter, Bodoni_Moda } from "next/font/google"; // Changed Playfair to Bodoni
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });
const bodoni = Bodoni_Moda({ 
  subsets: ["latin"], 
  variable: '--font-bodoni' // Use a clear variable name
});

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} ${bodoni.variable} bg-navy-900`}>
        {children}
      </body>
    </html>
  );
}
