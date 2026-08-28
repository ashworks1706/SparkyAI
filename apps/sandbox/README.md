# apps/sandbox

Phase 7. Isolated browser worker for authenticated, user-consented tasks. Separate container per session; called by the engine over a small task protocol; never linked into the API process. Engine (Playwright in Python or Node vs. chromiumoxide) decided when the phase starts. See `docs/ARCHITECTURE.md`.
