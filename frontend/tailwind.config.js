/** @type {import('tailwindcss').Config} */
export default {
  // This content array tells Tailwind which files to scan for classes.
  // This is the most critical part of the config.
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}