---
name: load
title: Session Load Workflow
description: Session lifecycle management with Serena MCP for project context loading, activation, and checkpoint restoration.
category: session
estimatedTime: 5-15 minutes
requiredApprovals:
  - plan-review
agents:
  - agent
---

## When to use

- Session initialization with prior context to restore.
- Cross-session persistence and memory retrieval needs.
- Project activation and context management requirements.
- Checkpoint loading and session continuity scenarios.

## Triggers

- User requests to load or restore a session/project context.
- Starting work on a project with prior checkpoints or memories.
- Need to rehydrate dependencies, configs, or prior decisions.

## Usage

- `/sc:load [target] [--type project|config|deps|checkpoint] [--refresh] [--analyze]`

## Behavioral flow

1. **Initialize**: Establish connection for session context management.
2. **Discover**: Inspect project structure and determine loading needs.
3. **Load**: Retrieve memories, checkpoints, configs, and dependencies context.
4. **Activate**: Set session context and prep for workflow execution.
5. **Validate**: Confirm loaded context integrity and readiness.

## MCP integration

- Serena MCP for memory retrieval, checkpoint loading, and session management.
- Performance targets: <500ms initialization; <200ms core memory operations.

## Tool coordination

- Project activation to establish working context.
- list/read memory for retrieval and restoration.
- Read/search for config/deps analysis when requested.
- Write operations for documenting active session state.

## Key patterns

- Project activation: directory analysis → memory retrieval → context establishment.
- Session restoration: checkpoint load → validation → workflow prep.
- Memory management: cross-session continuity → efficiency → readiness.
- Performance focus: fast initialization and validation.

## Examples

- Basic load: `/sc:load` (activates current project with memory integration)
- Specific project: `/sc:load /path/to/project --type project --analyze`
- Checkpoint restore: `/sc:load --type checkpoint --checkpoint session_123`
- Dependency refresh: `/sc:load --type deps --refresh`

## Boundaries

**Will:**

- Load project context with integrated memory management.
- Provide session lifecycle management with persistence and checkpoints.
- Establish project activation with validation.

**Will Not:**

- Modify project structure or configs without explicit intent.
- Operate without validated memory access.
- Override existing context without checkpoint preservation.

## Session continuity

- Start with `/sc:load` to restore context/checkpoints.
- Finish with `/sc:save` to persist progress and checkpoints.
