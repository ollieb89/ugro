---
name: save
title: Session Save Workflow
description: Session lifecycle management with Serena MCP for context persistence, checkpoints, and cross-session continuity.
category: session
estimatedTime: 1-2 minutes
requiredApprovals:
  - plan-review
agents:
  - agent
auto_execution_mode: 3
---

## When to use
Always start with activate_project to ensure the project is loaded.

- Session completion with context to preserve for later continuation.
- Cross-session memory management and checkpoint creation.
- Project understanding preservation and discovery archival.
- Progress tracking for complex or long-running tasks.

## Triggers

- User requests to save or checkpoint a session.
- Session exceeds 30 minutes or completes a major milestone.
- Significant discoveries, patterns, or decisions made.

## Usage

- `/sc:save [--type session|learnings|context|all] [--summarize] [--checkpoint]`

## Behavioral flow

1. **Analyze**: Review session progress; identify discoveries, decisions, and risks.
2. **Persist**: Store session context, learnings, and artifacts via Serena MCP memory.
3. **Checkpoint**: Create recovery points for complex sessions and progress tracking.
4. **Validate**: Confirm data integrity and cross-session compatibility.
5. **Prepare**: Note next steps and ensure readiness for seamless resume.

## MCP integration

- Serena MCP for memory management, checkpoints, and cross-session persistence.
- Performance targets: <200ms for memory ops, <1s for checkpoint creation.

## Tool coordination

- write/read memory operations for context persistence and retrieval.
- Session summarization for concise checkpoints.
- Todo status read to trigger automatic checkpoints when tasks complete.

## Key patterns

- Session preservation: analyze → persist → checkpoint → validate.
- Cross-session learning: accumulate context and patterns for future efficiency.
- Progress tracking: automatic checkpoints after milestones or time thresholds.
- Recovery planning: keep validated checkpoints ready for restoration.

## Examples

- Basic save: `/sc:save` (stores session context; auto-checkpoint if long-running)
- Comprehensive checkpoint: `/sc:save --type all --checkpoint`
- Summary-focused: `/sc:save --summarize`
- Learnings-only: `/sc:save --type learnings`

## Boundaries

**Will:**

- Save session context with Serena MCP integration.
- Create automatic checkpoints based on progress and task completion.
- Preserve discoveries, patterns, and decisions for cross-session continuity.

**Will Not:**

- Operate without validated memory integration.
- Overwrite existing context without checkpointing.
- Skip integrity validation for saved data.

## Session continuity

- Start with `/sc:load` to restore context/checkpoints.
- Finish with `/sc:save` to persist progress and checkpoints.
