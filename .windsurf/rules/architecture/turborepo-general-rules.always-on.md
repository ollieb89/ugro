---
trigger: always_on
---

# Turborepo Agent Rules

## Purpose
Turborepo coordinates builds, tests, and tooling across a monorepo with caching and dependency awareness.

---

## Mandatory Usage
Agents MUST:
- Use `apps/*` for apps and `packages/*` for shared code
- Define all runnable tasks in `turbo.json`
- Declare `outputs` for cacheable tasks
- Model dependencies using `dependsOn`

---

## Prohibited Usage
Agents MUST NOT:
- Use nested packages
- Import across packages via relative paths
- Run long-lived tasks with caching enabled

---

## Task Rules
- `build`: cacheable, depends on `^build`
- `dev` / `watch`: `persistent: true`, `cache: false`

---

## Completion Rule
Configuration is complete when tasks are deterministic and cache-safe.
