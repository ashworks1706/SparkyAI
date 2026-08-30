# apps/web

Public landing site. Vite + React + TypeScript + Tailwind + shadcn/ui.

```bash
cd apps/web
npm ci
npm run dev       # http://localhost:5173
npm run lint
npm test
npm run build     # dist/
```

Deployed as static files (Vercel or any static host). No runtime dependency on the engine; the admin UI (Phase 4) will live here too and talk to the engine over HTTP.
