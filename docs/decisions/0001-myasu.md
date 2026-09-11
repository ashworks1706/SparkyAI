# MyASU moves into scope, read-only first

2026-09-05.

MyASU was listed under "out of scope until stated otherwise". It moves into phase 8, which
already covers authenticated browser sessions through the Playwright MCP server.

Read-only first: hours, holds, deadlines, enrolled sections. No submission, no payment, no form
that changes a record until the read path has run in front of real students.

What does not move: GPA and coursework access, and FERPA claims. Both stay out of scope.
Reading a page a student can already see is not the same as holding academic records, and
saying so in public is a separate decision with its own obligations.

The invariants in ARCHITECTURE.md already cover the mechanism and stay as they are: one
isolated browser context per user, the user completes login and MFA themselves, SparkyAI never
asks for or stores a password, authenticated page content is never indexed, memorized, or
written to a trace, and any consequential submission is confirmed immediately before it runs.
