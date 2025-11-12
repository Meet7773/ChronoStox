import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// 1. Import the new tailwindcss/vite plugin
import tailwindcss from '@tailwindcss/vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // 2. Add the tailwindcss plugin to the array
    tailwindcss(),
  ],
})