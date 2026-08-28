# apps/sandbox

Phase 7. Isolated browser worker for authenticated, user-consented tasks. Python + Playwright, one container per session, called by the engine over a small HTTP task protocol. Never linked into the engine; never exposed publicly.

```bash
cd apps/sandbox
uv sync --extra dev
uv run playwright install chromium
uv run sandbox serve
```

| Module | Responsibility |
|---|---|
| `protocol.py` | `BrowserTask` / `BrowserResult` — the contract with `engine/src/agent/harness/sandbox.rs` |
| `server.py` | FastAPI: run task, status, confirm, end session |
| `session.py` | one Playwright context per user; user does login/MFA; encrypted state; expiry |
| `actions/` | navigate, read, fill (prepare), submit (needs confirmation token) |
| `policy/` | domain allowlist, download quarantine, time/step/size limits |
| `observe.py` | page → structured, size-limited observation |
| `redact.py` | secrets never reach logs, screenshots, or the engine |
| `artifacts.py` | redacted screenshots and action logs with TTL |

Rules (from `docs/ARCHITECTURE.md`): the user completes login and MFA themselves; the sandbox never asks for or stores a password; authenticated page content is never indexed or memorized; every submit/book/post/delete requires a confirmation bound to the exact payload; CAPTCHA, MFA failure, expired session, or an unexpected page stops the task.

Requires explicit authorization before implementation begins (see roadmap out-of-scope).
