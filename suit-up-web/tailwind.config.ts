// tailwind.config.ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          900: "#020617", // Deepest Navy
          800: "#0f172a", // Dark Navy
        },
        gold: {
          400: "#FBDF9D", // Light Gold
          500: "#C5A059", // Classic Gold
          600: "#A3803F", // Deep Gold
        }
      },
      fontFamily: {
        serif: ['var(--font-playfair-display)', 'serif'], // Elegant serif
      }
    },
  },
  plugins: [],
};
export default config;
