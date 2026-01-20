---
trigger: always_on
---

# Turborepo + Prisma Rules

## Purpose
Manage database schema and client generation safely.

---

## Mandatory Usage
- Prisma lives in a dedicated DB package
- Client generation is task-driven

---

## Prohibited Usage
- Caching migrations

---

## Tasks
- `db:generate`: cacheable
- `db:migrate`: non-cacheable
