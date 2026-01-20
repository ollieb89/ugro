---
trigger: always_on
---

# Turborepo + Vite Rules

## Purpose
Ensure Vite apps integrate cleanly with Turborepo caching and task orchestration.

---

## Mandatory Usage
- Vite apps live in `apps/*`
- Build outputs (`dist/**`) must be declared
- Shared config lives in a package

---

## Prohibited Usage
- Cross-app imports
- Cached dev servers

---

## Tasks
- `build`: cacheable
- `dev`: persistent, non-cacheable
