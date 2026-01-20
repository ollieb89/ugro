---
trigger: always_on
---

# Turborepo + TypeScript Rules

## Purpose
Standardize TypeScript configuration and builds.

---

## Mandatory Usage
- Use shared TS config packages
- Packages extend shared configs

---

## Prohibited Usage
- Root-level `tsconfig.json`

---

## Tasks
- `build`: cacheable
- `typecheck`: cacheable
