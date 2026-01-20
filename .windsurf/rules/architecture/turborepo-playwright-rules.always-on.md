---
trigger: always_on
---

# Turborepo + Playwright Rules

## Purpose
Run end-to-end tests reliably with dependency awareness.

---

## Mandatory Usage
- Isolate Playwright in a test package
- E2E tests depend on `build`

---

## Prohibited Usage
- Caching watch mode

---

## Tasks
- `test:e2e`: cacheable
- `test:e2e:watch`: persistent
