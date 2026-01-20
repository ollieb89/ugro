---
trigger: always_on
description: This constitution defines the **mandatory operating rules** for agents using the Model Context Protocol (MCP). It integrates **reasoning**, **documentation**, **memory**, **build orchestration**, and **UI standards** into a single, enforceable framew
---

# MCP Constitution (Full)

---

## Description

This constitution defines the **mandatory operating rules** for agents using the Model Context Protocol (MCP).
It integrates **reasoning**, **documentation**, **memory**, **build orchestration**, and **UI standards** into a single, enforceable framework.
---

## 1. Sequential Thinking (Reasoning)

### Purpose

Provide explicit, auditable reasoning for complex decisions.

### Mandatory Usage

Agents MUST use Sequential Thinking for:

- Planning or re-planning work
- Architectural or design decisions
- Dependency and risk analysis
- Complex bugs or blockers
- High-risk or irreversible actions

### Prohibited Usage

- Trivial execution
- Post-hoc justification
- Documentation lookup

### Completion Rule

Reasoning is complete only when a plan, decision, or resolution artifact exists.

---

### 2. Context7 (Documentation)

## Purpose Context7

Retrieve current, authoritative documentation.

## Mandatory Usage of Context7

Agents MUST invoke Context7 when:

- APIs or versions are uncertain
- Migrations or upgrades are involved
- Best practices affect correctness or safety

## Prohibited Usage of Context7

- General reasoning
- Trivial lookups
- Copying large doc sections

### Ordering Rule

**Sequential Thinking → Context7 → Execution

---

## 3. Memory & Checkpointing

### Purpose of Memory & Checkpointing

Preserve continuity across actions and sessions.

### Mandatory Usage of Memory & Checkpointing

Agents MUST write memory when:

- Plans or decisions change
- Blockers are found or resolved
- Milestones are reached
- Sessions end with open work

### Prohibited Usage of Memory & Checkpointing

- Raw chain-of-thought
- Documentation dumps
- Trivial logs

### Completion Rule of Memory & Checkpointing

State must be resumable without loss of intent.

---

## 4. Turborepo (Build & Task Orchestration)

### Purpose of Turborepo

Coordinate monorepo builds, tests, and tooling with caching and dependency awareness.

### Mandatory Usage of Turborepo

- Use `apps/*` and `packages/*`
- Define all tasks in `turbo.json`
- Declare outputs for cacheable tasks
- Model dependencies with `dependsOn`

### Prohibited Usage of Turborepo

- Nested packages
- Cross-package relative imports
- Cached long-running tasks

---

## 5. UI Rules (Tailwind CSS + shadcn/ui)

### Purpose of UI Rules

Ensure consistent, accessible, and scalable UI.

### Tailwind CSS

- Utility-first styling is mandatory
- Design tokens live in Tailwind config
- Mobile-first responsive design

### shadcn/ui

- Components are source code, not a library
- Accessibility guarantees must be preserved
- Shared components belong in a UI package

### Prohibited Usage of UI Rules

- Custom CSS for layout already covered by Tailwind
- Inconsistent component forks
- Accessibility regressions

---

## 6. Global Anti-Patterns

- Hidden reasoning
- Stale documentation usage
- Lossy session endings
- Task orchestration without dependency modeling
- UI inconsistency across apps

---

## 7. Constitutional Completion Rule

An agent action is considered valid only if:

- Reasoning is explicit where required
- Documentation is current where needed
- State is persisted when necessary
- Builds and tasks are deterministic
- UI remains consistent, responsive, and accessible