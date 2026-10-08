# apps/web

Public landing site. Vite, React, TypeScript, Tailwind 4. Self-hosted fonts. Runtime dependencies: React, React Router, motion, lenis and lucide-react.

```bash
just web          # dev server on :5173
just check-web    # eslint, tsc, vitest, build
```

The demo videos in `public/demo` are rendered from `src/demo` frame by frame: `CHROMIUM=/path/to/chrome npm run demo:record` (needs ffmpeg).

The build is static with no runtime engine dependency. The Phase 4 admin UI will call the engine over HTTP. Deployment: see [deploy/README.md](../../deploy/README.md).
