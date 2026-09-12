# 0005 — Drop the browser MCP server

2026-09-11

## Decision

SparkyAI ships no browser. The Playwright MCP server, its compose service, its `just browser`
recipe, and the `mcp.playwright_url` setting are removed. The generic MCP client stays: a server
configured under `[[mcp.servers]]` still becomes tools.

## Why

Browser tools were the widest part of the attack surface and the least used part of the answer
path. Every ASU page the assistant answers from is already ingested by the scraper or fetched by
`query_source`, so the browser earned nothing on the read side, and the write side it enabled
was held behind confirmation anyway.

## Consequences

Phase 8 of the roadmap no longer has a mechanism. Authenticated tasks, MyASU included, need a
new decision before any work starts; [0001-myasu](0001-myasu.md) assumed this server and is
superseded on that point. Nothing in the current answer path changes.
