---
name: SSH agent forwarding is ephemeral; commit locally when user disconnects
description: Agent-forwarded SSH only works while the user is actively connected. If SSH auth fails during autonomous work, commit locally and defer the push — do not try to diagnose or switch to a different remote.
type: feedback
originSessionId: 18187ffd-2357-464e-8e2b-b0792064cab7
---
SSH is agent-forwarded from the user's laptop. When the user SSHes in, a fresh `/tmp/vscode-ssh-auth-sock-*` is created and `SSH_AUTH_SOCK` points at it. When the user disconnects, the socket becomes stale and git push fails with `Permission denied (publickey)`.

**Why:** only the user has the private key; it's not on the machine. Attempting to push when they're disconnected will always fail no matter how I manipulate `SSH_AUTH_SOCK`.

**How to apply:**
- If a push fails with `Permission denied (publickey)`, don't spelunk — assume the user disconnected. Continue committing locally; the commits wait on main.
- When the user reconnects, their new SSH_AUTH_SOCK is under `/tmp/vscode-ssh-auth-sock-*` (most recent mtime). Re-export it and push is immediate.
- If I'm uncertain whether they're connected, I can just attempt the push; failure is harmless.
