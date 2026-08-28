"""Wire types shared with the engine's `agent::harness::sandbox`.

BrowserTask (session_id, allowlist, steps, deadline) -> BrowserResult (observations,
proposed_action, receipt, status).

Keep this file in sync with the Rust side; it is the contract.
"""
