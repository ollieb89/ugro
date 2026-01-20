---
trigger: always_on
description: Sequential Thinking usage rules for agents, enforced globally.
---

# Sequential Thinking Agent Rules (MCP)

## Purpose
Sequential Thinking is used to externalize structured, step-by-step reasoning for complex decisions, planning, and risk-sensitive operations.

Agents must use this tool only when reasoning materially improves correctness, safety, or clarity.

---

## Mandatory Usage

Agents MUST invoke Sequential Thinking when performing any of the following:

- Creating or revising plans, phases, or task hierarchies
- Making architectural or design decisions with tradeoffs
- Performing dependency analysis or risk assessment
- Investigating complex bugs or performance issues
- Evaluating multiple viable solution options
- Diagnosing or resolving blockers
- Before executing high-risk, destructive, or irreversible actions

---

## Prohibited Usage

Agents MUST NOT use Sequential Thinking for:

- Trivial, mechanical, or purely procedural steps
- Straightforward execution of an agreed plan
- Documentation lookup or summarization
- Post-hoc justification of decisions already made

Overuse is considered an anti-pattern.

---

## Invocation Rules

When Sequential Thinking is invoked, the agent must:

1. Explicitly state the reasoning goal
2. Constrain scope to the immediate problem
3. Reason step-by-step without skipping assumptions
4. Consider at least one alternative when making decisions
5. Reach a decision or explicitly document deferral

---

## Required Outputs

Every Sequential Thinking session MUST produce at least one durable artifact:

- `plan_*` — structured plans or decompositions
- `decision_*` — architectural or design choices
- `blocker_*` — root-cause analysis
- `resolution_*` — selected solution and rationale
- `checkpoint_*` — updated state snapshot

Reasoning without artifacts is considered incomplete.

---

## Tool Ordering Rule

Sequential Thinking MUST precede other MCP tools:

**Sequential Thinking → Context7 → Execution**

Reasoning determines what documentation is needed, not the reverse.

---

## Safety and Auditability

- Clearly separate facts, assumptions, and judgments
- Record uncertainty explicitly
- Avoid reasoning toward a predetermined outcome
- Ensure conclusions are proportional to evidence

If reasoning affects safety, data integrity, or user trust, it must be recorded.

---

## Anti-Patterns (Disallowed)

- Hidden or implicit reasoning
- Verbosity without decisions or outcomes
- Using Sequential Thinking as a memory store
- Mixing execution steps into reasoning output

---

## Completion Rule

A Sequential Thinking session is complete only when:

- A plan, decision, or resolution is recorded, OR
- The problem is explicitly deferred with documented reasons and next steps.
