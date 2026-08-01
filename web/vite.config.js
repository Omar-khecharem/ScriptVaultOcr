import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Le proxy /api redirige les requêtes vers le backend FastAPI local
// (python main.py, par défaut http://127.0.0.1:8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.SCRIPTVAULT_API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2020",
  },
});
