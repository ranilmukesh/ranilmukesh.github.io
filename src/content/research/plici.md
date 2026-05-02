---
title: "Signed and Sealed: Protocol-Level Isolation with Cryptographic Integrity (PLICI) for Secure LLM Agents"
pdfUrl: "/papers/PLICI-flatten.pdf"
abstract: "PLICI is a deterministic security middleware for LLM agents that shifts defense away from fragile prompt-level heuristics toward verifiable protocol enforcement. It uses session-scoped Ed25519 signatures to verify data origin, schema-locked enclaves to isolate tool payloads before the agent reads them, and an NLI-based semantic filter to catch logic hijacking, sharply reducing attack success while keeping latency low enough for real-time agent systems."
tags: ["LLM Security", "Prompt Injection", "Cryptography", "MCP", "Agentic AI"]
---

The main idea is simple but strong: agents should not trust tool outputs just because they arrive in the workflow, and every external payload should prove where it came from and what parts are safe to expose. That makes this paper particularly relevant for MCP-style multi-agent systems, enterprise tool use, and any high-stakes workflow where indirect prompt injection is a serious risk.
