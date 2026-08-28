---
name: code-quality
description: >-
  Run a disciplined quality pass over existing code: naming and structure for
  longevity, reuse of what the workspace already depends on instead of
  hand-rolling, collapsing enumerated one-off units into data-driven
  registries, and removing dead code, silent fallbacks, mock data, and unused
  scaffolding. Use when asked to clean up, refactor, de-duplicate, harden,
  make production-ready, reduce tech debt, reorganize or rename, or review code
  for quality. Trigger on "clean this up", "make this production-ready",
  "make it more modular", "reduce duplication", "get rid of the dead code",
  "we already use X, use that", or "is this good code?".
disable-model-invocation: true
---


# Code quality pass

A quality pass is mostly **subtraction and consistency**, not cleverness. The
goal is a codebase that is easy to read, safe to change, honest about what it
actually does, and **built to survive growth** — a new engineer should be able
to open it and understand what's going on. You are not adding features here —
you are making the existing code better at being itself.

Ground the structural calls in established modularity principles — high
cohesion, low coupling, separation of concerns, one reason to change per module
— rather than personal taste. When the right seam isn't obvious, reason from
those, not from aesthetics.

## Prime directive

**Preserve behavior — unless the behavior is itself the bug.** Most changes here
must leave observable behavior identical. The one deliberate exception is code
that *hides* failure (silent fallbacks, mock data, swallowed errors): there, the
current behavior is the bug, and making it fail honestly is the fix. Be explicit
with yourself about which case you're in before you touch anything.

## Workflow

Work in small, verifiable units — never one giant sweeping rewrite.

1. **Map before you touch.** Read the folder. Find what's actually used: grep
   for every reference to a symbol/file across the whole tree (including dynamic
   or string-based references) before assuming anything is dead. Identify the
   real public surface vs. internals. **Also read the manifests** (root `Cargo.toml`
   `[workspace.dependencies]`, the crate's `Cargo.toml`, each Python
   package's `pyproject.toml`, `apps/web/package.json`) and `AGENTS.md` / `docs/ARCHITECTURE.md` so you
   know what the project already has and what its boundaries are before you
   keep hand-rolling around them.
2. **Change one concern at a time.** Rename, then stop. Extract, then stop. Each
   step should leave the code working.
3. **Prove it still works.** Run `/check` (fmt, clippy `-D warnings`, tests,
   dependency-direction check) after each meaningful change. If there are no
   tests for the affected path, exercise it and say so.
4. **Migrate callers, then delete.** When renaming or moving, update every
   reference first; only delete the old thing once nothing points at it. This
   applies to library adoption too: migrate the call sites onto the library,
   *then* delete the hand-rolled helper.

If a change would touch many callers or restructure folders, confirm with the
user before doing it — don't silently reshape their tree.

---

## 1. Professionalism — naming & structure for longevity

Names and layout should survive growth. A good name describes a thing's **role**,
not today's incidental **instance**. Optimize for the codebase this will *become*
as more of the same kind of thing gets added, not just the one that exists now.

**Naming**
- Prefer role over instance. `fetch_user` scales; `fetch_user_from_postgres_v2`
  bakes in a vendor and a version that will rot. `parse_html` scales;
  `parse_asu_library_page` hardcodes one caller.
- **Name the category, not each instance — scale by registration, not
  enumeration.** When a system grows by adding more of the *same kind* of thing
  (tools, ingestion sources, scorers, policies, providers, steps), model that
  kind **once** as a trait and let the instances be **data** — a registry, a
  `Vec<Box<dyn Tool>>`, a map keyed by name — that the code iterates or looks
  up. A `ToolRegistry` you iterate scales; a wall of `run_library_tool`,
  `run_events_tool`, `run_clubs_tool`… enumerated one-per-symbol does not (v1
  had eleven copy-pasted agents for exactly this reason). A name that bakes one
  instance into a generic role (`LibraryToolFn`, `planner_for_the_query`) is a
  signal the abstraction is too narrow: the thing wants to be a general
  `Tool` / `Source` selected by config, with the specific case as an entry.
  This is the single biggest lever for scalability in a folder that grows by
  accretion.
- Kill magic values baked into names and bodies. A number or string with meaning
  should be a named constant, defined once.
- Be consistent: one concept → one word, everywhere. Don't let `user`,
  `account`, and `profile` refer to the same thing in three files.
- But don't over-abstract into meaninglessness. `data`, `handler`, `manager`,
  and a `utils` that does everything are just as bad as over-specific names. Aim
  for **general enough to scale, specific enough to be unambiguous.**

**File & folder naming**
- Files are named for their **domain/role**, not one caller or one instance.
  Prefer `sources/` with one entry per source over `library_scraper_v2.rs`.
  The same narrowness that hurts function names hurts filenames, and renaming
  files is more disruptive, so it rots harder.
- Once a folder establishes context, don't repeat it in the filename — a file
  inside `sources/` doesn't need `source` in its own name. Let the path
  carry the qualifier so names stay short and the domain reads cleanly.
- **One casing convention across the tree.** Rust: `snake_case` modules and
  files, `CamelCase` types, `SCREAMING_SNAKE` consts. Python: PEP 8. Don't mix
  for peers in the same layer.

**Structure**
- Group by feature/domain where that aids scale, so a new contributor can *guess*
  where a file lives without a map.
- Keep folders shallow and predictable — but also watch the opposite failure: a
  folder that has accreted dozens of sibling files is a signal to group by
  sub-domain. Only do so when a *natural* grouping exists; don't invent arbitrary
  buckets to hit a file count.
- Restructure only when the current layout actively causes friction. Reshuffling
  for aesthetics costs churn and buys nothing — leave a working layout alone.

## 2. Upcycling — modularize & reuse

Reuse what exists; split what has outgrown itself.

**Modularize big files**
- Split a file when it *mixes responsibilities* or you have to scroll to
  understand it — not merely because it's long. Split along the natural seams so
  each module does one thing.
- After splitting, the pieces should be nameable in a sentence each. If they
  aren't, the seam was wrong.

**Reuse over recreate (in-repo)**
- Before writing a function, search for one that already does the job or nearly
  does it. Extend it rather than cloning it.
- Two near-duplicate functions → unify into one parameterized function **only if
  the abstraction is honest.** The rule of three: duplication once or twice is
  fine; extract when the pattern is genuinely repeated. Forcing unrelated code
  together under one function is worse than the duplication — the wrong
  abstraction is expensive to undo.
- **N near-identical named units that differ only by a constant → one unit
  driven by that value, registered in a collection.** This is the reuse form of
  "name the category, not each instance" (§1): the same predicate copied across
  `run_library_tool` / `run_events_tool` wants to be one `run_tool(&dyn Tool)`
  iterated over the registry. Honesty still applies — only unify
  cases that are truly the same shape; a registry of things that aren't really
  alike is just a switch statement wearing a costume.

**Reuse the dependencies the project already has**

The most common form of accidental duplication isn't two copies of a function —
it's a hand-rolled version of something already sitting in the manifest. Before
keeping any bespoke helper, check whether an installed dependency covers it.

- **Check `[workspace.dependencies]` first.** The workspace already has an HTTP
  client (`reqwest`), HTML parser (`scraper`), config loader (`figment`),
  error derive (`thiserror`), ids (`uuid`), time (`chrono`), model client and
  tool schema (`rig-core`), MCP (`rmcp`). Bespoke versions of those jobs are
  cruft. Migrate the callers, then delete the hand-rolled version.
- **Follow the convention the project already chose.** `tracing` not
  `println!`; `thiserror` enums not `anyhow` in library crates; `SecretString`
  for secrets; config through `crates/app/src/config.rs` only. One concept, one
  tool.
- **Prefer the standard library.** A hand-rolled helper *or* a stale dependency
  doing what `std` already does can be deleted outright — the best reuse adds
  nothing.
- **Delete the now-unused dependency.** Once nothing imports it, remove it from
  the crate's `Cargo.toml` and, if no crate uses it, from
  `[workspace.dependencies]`. A dead dependency is dead code with a CVE feed.

**Introducing a *new* library — treat as an addition, not a cleanup**

Replacing 400 lines of hand-rolled parsing logic with a well-maintained
library is often the single highest-value change in a quality pass. But adding a
dependency is the one move in this skill that *grows* the project's surface, so
it needs a higher bar than the rest.

- **Confirm with the user before adding any new dependency.** Never add one
  silently during a cleanup pass.
- **Justify it by what it deletes.** Name the concrete lines/files it removes and
  the correctness problems it fixes (edge cases, timezones, escaping,
  encoding). If it doesn't delete meaningful code or fix real
  bugs, don't add it.
- **Weigh the real cost**: install/bundle size, maintenance status and release
  cadence, transitive dependency count, license, and how hard it would be to back
  out. Prefer boring, widely-adopted, actively-maintained options over clever
  ones.
- **Don't import a library for something trivial.** A well-understood 5–20 line
  helper is cheaper to own than a dependency to track. Reach for a library when
  the domain is genuinely hairy — dates/timezones, parsing, cryptography,
  sanitization — not to avoid writing a `clamp()`.
- **Adopt incrementally.** Migrate one module or one surface, verify, then
  continue. Half-migrated is acceptable at a commit boundary only if it's
  deliberate and stated; two competing systems left permanently side by side is
  worse than the original mess.

## 3. Cleaning — remove the cruft

Delete what nobody needs. Every line you remove is a line no one has to read,
trust, or maintain.

**Silent fallbacks that mask bugs**
- Watch for `.unwrap_or_default()`, `.ok()` that discards the error, `let _ =`
  on a `Result`, `if let Ok(x)` with no `else`, or a `match` arm that returns a
  default on error. These convert a loud bug into an invisible wrong answer.
  (v1's `except Exception: return "No results"` pattern is the cautionary tale.)
- Remove the mask so the failure surfaces (this flows into Fidelity below).
- Keep *intentional* defaults — a documented config default, a real fallback
  strategy. The target is only the ones that exist to make errors disappear.

**Unnecessary optionality**
- Options nobody sets, flags always passed the same value, config knobs with a
  single caller: collapse to the one path that's actually used.

**Stale / duplicate / dead code**
- Unreferenced functions, commented-out blocks, unreachable branches, and old
  versions left sitting beside the new one.
- Unused dependencies in a `Cargo.toml`, and duplicate dependencies that do
  the same job (two HTTP clients, two error crates) — consolidate onto one.
- Verify truly unused across the whole tree (including dynamic references)
  **before** deleting. Prove it's dead; don't guess.

## 4. Fidelity — fail honestly, no fakes

Shipped code should tell the truth. A loud failure is debuggable; a silent wrong
answer is not.

**Remove mock data and fake displays from real paths**
- Hardcoded sample responses, stubbed returns that pretend an integration
  works, a `ready()` that returns 200 without checking anything — in production
  paths these lie to the user and hide missing wiring. Remove them or make them
  fail loudly.
- Keep legitimate test fixtures and the `mock` model provider. The target is
  fakes *masquerading as real behavior in shipped code*, not honest test
  scaffolding.

**Fail faithfully**
- When data is missing, a fetch fails, or input is bad, surface it — return the
  typed error, `tracing::error!` with fields — instead of returning a default
  that merely *looks* fine. Don't let the system report success it didn't earn.

**Remove unused pass-throughs**
- `pub use` re-exports and wrapper functions that nothing outside the crate
  consumes add indirection and hide the real module. Delete the ones with no
  consumers.
- Keep re-exports that form a real, *used* public surface — those earn their
  place.

---

## Guardrails (read before deleting or restructuring)

- **Behavior is preserved unless the behavior is the bug.** Know which case
  you're in.
- **Small units, verified.** `/check` after each meaningful change.
- **Fake vs. defensive.** An intentional default or a genuine fallback strategy
  is not the enemy — only the ones that hide failure are.
- **Duplication beats the wrong abstraction.** Don't unify things that aren't
  really the same — including into a dynamic registry. Scale by registration
  only when the instances share a real shape.
- **Prove "unused" before deleting.** Grep the whole tree, including dynamic and
  string references — this matters doubly for registries and config-keyed tables,
  where references are string-based and easy to miss.
- **Prefer, in order:** delete it → the platform/standard library → a dependency
  the project already has → a small owned helper → a new dependency (with the
  user's agreement).
- **Never add a dependency silently.** New dependencies are additions and must be
  confirmed, justified by what they delete, and adopted incrementally.
- **Confirm wide-blast changes.** Renames or moves that touch many callers, any
  folder restructure, any file-renaming sweep, and any migration onto a
  different library get confirmed with the user first.
- **Readable by a new engineer.** The end state a stranger can navigate beats a
  clever one only the author understands. If a rename or restructure doesn't move
  toward that, don't do it.
- **Respect crate boundaries.** A cleanup never moves code across the
  `app → adapters → harness` direction or makes adapters import each other;
  `scripts/check-deps.sh` will reject it anyway.

## Done means

The folder reads more clearly than before and a new engineer could find their
way around it; names describe roles and scale by registration rather than
enumerating one symbol per case; files and folders are domain-focused with
consistent casing and no over-stuffed directories; nothing behaves differently
except failures that were previously hidden; every deletion is provably unused;
hand-rolled code that duplicated an available library has been migrated and
deleted; no dependency was added without the user's agreement; and `/check` is
green.