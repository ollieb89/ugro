---
description: This mode follows Model Context Protocol (MCP) guidance by requiring **explicit tool invocation**, **auditable reasoning artifacts**, and **clear separation between reasoning, documentation lookup, and execution**.
---

# Task Management Mode (MCP-Aligned)

## Purpose and Scope

Use this mode for multi-step or multi-phase work that benefits from explicit planning, documented reasoning, and tool-assisted execution using MCP-compatible servers.

This mode follows Model Context Protocol (MCP) guidance by requiring **explicit tool invocation**, **auditable reasoning artifacts**, and **clear separation between reasoning, documentation lookup, and execution**.

## When to Use

- Tasks with more than three dependent steps
- Work spanning multiple files, directories, or subsystems
- Phased execution, dependency tracking, or structured refinement
- Manual invocation via flags such as `--task-manage` or `--delegate`
- Quality, polish, refactor, or reliability improvements

## Core Operating Principles

- Hierarchical decomposition: **Plan → Phase → Task → Todo**
- Explicit reasoning via the **Sequential Thinking MCP server**
- Just-in-time documentation via the **Context7 MCP server**
- Continuity through explicit written artifacts (plans, checkpoints, decisions)

MCP tools are invoked intentionally and only when they add value.

## Task Hierarchy Model

- 📋 **Plan** — Overall objective, constraints, and success criteria
- 🎯 **Phase** — Major milestone or logical stage
- 📦 **Task** — Concrete deliverable within a phase
- ✓ **Todo** — Atomic, executable action

## MCP Tool Integration Strategy

### Sequential Thinking (MCP Server)

Use for:

- Planning and decomposition
- Dependency and risk analysis
- Architectural tradeoff evaluation
- Blocker diagnosis and resolution analysis

Summarize reasoning outputs into durable artifacts such as plans or decisions.

### Context7 (MCP Server)

Use for:

- Fetching current APIs, types, configuration, or migration guidance
- Validating assumptions about libraries or frameworks
- Capturing best practices relevant to the task

Record distilled findings rather than copying large documentation excerpts.

## Execution Workflow

### 1. Initialization

- Review existing artifacts (plans, checkpoints, decisions).
- If no plan exists, invoke Sequential Thinking to define:
  - Overall plan
  - Initial phases
  - Constraints and risks

### 2. Planning (Reasoning-First)

- Decompose work into phases, tasks, and todos.
- Use Sequential Thinking to justify structure, dependencies, and risks.
- Persist results as explicit planning artifacts.

### 3. Implementation Preparation

- Identify unknowns or version-sensitive areas.
- Use Context7 to retrieve relevant documentation.
- Extract actionable APIs, patterns, or constraints.
- Record summarized findings.

### 4. Execution

- Complete todos sequentially.
- Maintain hierarchy and scope discipline.
- After meaningful progress, update task status and record decisions.

### 5. Blocker Handling

- Use Sequential Thinking to analyze root causes, options, and tradeoffs.
- Use Context7 if documentation is required.
- Record blocker analysis and resolution.

### 6. Quality Gates

- Before execution: confirm dependencies and risks are addressed.
- After each task: validate objectives and log checks or tests.
- At milestones: reassess plan and record checkpoints.

### 7. Session End / Resume

- At session end:
  - Summarize completed work
  - Record open items and next steps
  - Capture decisions and lessons learned
- On resume:
  - Read plan and latest checkpoint
  - Restate current status
  - Continue with the next task

## Artifact Naming Conventions

- `plan_<topic>_<date>`
- `phase_<n>`
- `task_<phase>_<n>`
- `todo_<task>_<n>`
- `checkpoint_<timestamp>`
- `decision_<topic>`
- `blocker_<topic>`
- `resolution_<topic>`
- `pattern_<topic>`

## Tool Invocation Triggers

### Sequential Thinking

- Creating or revising plans or phases
- Architectural or design decisions
- Complex bugs or performance issues
- Risk assessment before irreversible actions

### Context7

- New or unfamiliar frameworks/libraries
- Uncertain APIs or version changes
- Configuration, testing, or migration needs

## Anti-Patterns

- Skipping explicit reasoning for complex work
- Implementing against assumed or stale documentation
- Over-fetching documentation for trivial steps
- Failing to record decisions or resolutions

## Expected Outputs

- Hierarchical plan with phases, tasks, and todos
- Documented reasoning and decisions
- Summarized documentation findings
- Validation notes per task and phase
- Final checkpoint with outcomes and learnings
