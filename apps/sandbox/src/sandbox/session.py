"""One isolated Playwright browser context per user session.

Lifecycle: create -> user completes login/MFA in the context -> tasks run -> expire -> cleanup.

Session secrets (cookies, storage state) are encrypted at rest in a separate namespace and never
leave this process.
"""
