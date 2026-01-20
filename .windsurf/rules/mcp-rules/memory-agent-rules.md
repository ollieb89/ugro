---
trigger: model_decision
description: # Memory & Checkpointing Agent Rules (Serena MCP)
---

# Memory & Checkpointing Agent Rules (MCP)

---

## Purpose

Memory and checkpointing are used to preserve continuity, decisions, and progress across agent actions and sessions.

Agents must record durable state explicitly rather than relying on implicit context.

---

## Mandatory Usage
Agents MUST write memory artifacts when:

- A plan, phase, or task is created or modified
- A significant decision or tradeoff is made
- A blocker is identified or resolved
- A phase or milestone is completed
- Progress would be lost without persistence
- A session is ending with unfinished work

---

## Prohibited Usage

Agents MUST NOT use memory for:

- Raw chain-of-thought or transient reasoning
- Copying documentation verbatim
- Storing trivial execution logs
- Replacing source-of-truth systems (e.g. code, tickets)

---

## Artifact Types
Approved memory artifacts include:

- `plan_*` — goals, scope, constraints
- `phase_*` — milestone definitions and status
- `task_*` — task-level progress and notes
- `checkpoint_*` — snapshot of current state
- `decision_*` — architectural or design choices
- `blocker_*` — impediments and root causes
- `resolution_*` — how blockers were resolved
- `pattern_*` — reusable practices

---

## Writing Rules
When writing memory, agents must:

1. Be concise and factual
2. Record rationale, not raw reasoning
3. Note assumptions and uncertainty
4. Update rather than duplicate when possible
5. Tie artifacts to concrete tasks or phases

---

## Checkpointing Rules
- Create a checkpoint at meaningful milestones
- Always checkpoint before ending a session
- Checkpoints must summarize:
  - Completed work
  - Open tasks
  - Next actions
  - Key risks or dependencies

---

## Anti-Patterns
- Using memory as scratch space
- Duplicating identical artifacts
- Storing speculative or abandoned ideas
- Failing to update outdated artifacts

---

## Completion Rule
Memory handling is complete when the current state
can be resumed without loss of intent or context.