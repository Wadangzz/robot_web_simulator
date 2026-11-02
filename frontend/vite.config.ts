import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';

// /api(REST) 와 /meshes(URDF·메시 정적) 는 FastAPI 백엔드(:8500)로 프록시.
// → 프론트는 같은 출처로 fetch, CORS·경로 고민 없음.
// (백엔드 포트 8500: 8000 은 다른 로컬 개발 서버가 점유)
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './app') }, // shadcn 별칭 (소스=app/)
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8500',
      '/meshes': 'http://localhost:8500',
      '/ws': { target: 'ws://localhost:8500', ws: true }, // 실시간 grab&push 스트림
    },
  },
});
