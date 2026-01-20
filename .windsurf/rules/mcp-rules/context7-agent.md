---
trigger: always_on
---

# Context7 Agent Rules (MCP)

## Purpose
Context7 is used to retrieve **current, authoritative documentation** for libraries, frameworks, and tools.

Agents must use Context7 to reduce reliance on stale knowledge or assumptions.

---

## Mandatory Usage

Agents MUST invoke Context7 when:

- Starting work with an unfamiliar library or framework
- APIs, types, or configuration details are uncertain
- Version-specific behavior may affect implementation
- Performing migrations or upgrades
- Best practices are required for correctness or safety

---

## Prohibited Usage

Agents MUST NOT use Context7 for:

- General reasoning or planning
- Trivial or well-known APIs
- Repeated lookups without new uncertainty
- Copying large documentation sections verbatim

---

## Invocation Rules

When invoking Context7, agents must:

1. Clearly identify the library or framework
2. Narrow the scope to the needed topic (API, config, migration, etc.)
3. Retrieve only the minimum documentation required
4. Prefer authoritative, high-signal sources

---

## Output Requirements

Context7 results must be:

- Summarized into actionable notes or patterns
- Referenced by library and version (if applicable)
- Linked to the task or decision they support

Context7 outputs are inputs to execution, not final answers.

---

## Tool Ordering Rule

Context7 MUST follow reasoning:

**Sequential Thinking → Context7 → Execution**

---

## Anti-Patterns

- Over-fetching documentation
- Treating docs as reasoning
- Using Context7 to replace planning
- Failing to record findings

---

## Completion Rule

A Context7 session is complete when:
- Relevant documentation is summarized, AND
- Findings are recorded for the task at hand
