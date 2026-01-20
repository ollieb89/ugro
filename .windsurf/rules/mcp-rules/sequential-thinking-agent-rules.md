---
trigger: always_on
---

# Sequential Thinking Agent Rules (MCP)

## Purpose
Sequential Thinking is used to externalize structured, step-by-step reasoning for complex decisions, plans, and risk-sensitive work.

Agents must use this tool only when reasoning materially improves correctness, safety, or clarity.
---

## Mandatory Usage
Sequential Thinking MUST be used for:
- Planning or re-planning work
- Architectural or design decisions
- Dependency and risk analysis
- Complex bugs or performance issues
- Evaluating multiple solution options
- Blocker analysis
- High-risk or irreversible actions

---

## Prohibited Usage
Do NOT use Sequential Thinking for:
- Trivial or mechanical steps
- Straightforward execution
- Documentation summarization
- Justifying decisions already made

---

## Invocation Rules
1. Clearly state the reasoning goal
2. Constrain scope to the immediate problem
3. Reason step-by-step
4. Evaluate alternatives
5. Reach a decision or explicitly defer

---

## Output Requirements
All Sequential Thinking sessions must produce an artifact:
- plan_*
- decision_*
- blocker_*
- resolution_*
- checkpoint_*

Uncaptured reasoning is invalid.

---

## Tool Ordering
Sequential Thinking → Context7 → Execution

---

## Anti-Patterns
- Hidden reasoning
- Verbosity without decisions
- Using reasoning as memory
- Mixing execution into reasoning

---

## Completion Rule
A session ends only when a decision, plan, or deferral is recorded.
