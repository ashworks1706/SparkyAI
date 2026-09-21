# apps/web

Public landing site. Vite, React, TypeScript, Tailwind, shadcn/ui.

```bash
just web          # dev server on :5173
just check-web    # eslint, tsc, vitest, build
```

The build is static with no runtime engine dependency. The Phase 4 admin UI will call the engine over HTTP. Deployment: see [deploy/README.md](../../deploy/README.md).
