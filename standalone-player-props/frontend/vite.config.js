import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// When embedded in Flask at /player-props, production chunks must load from
// /player-props/assets/... (absolute /assets/... would 404 and break html2canvas).
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/player-props/" : "/",
  plugins: [react()],
  server: {
    port: 5179,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8101",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "") || "/",
      },
    },
  },
  preview: {
    port: 4179,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8101",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "") || "/",
      },
    },
  },
}));
