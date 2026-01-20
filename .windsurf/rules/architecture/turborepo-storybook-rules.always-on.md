---
trigger: always_on
---

# Turborepo + Storybook Rules

## Purpose
Develop and document UI components consistently.

---

## Mandatory Usage
- Storybook lives in UI/docs packages
- Build output is declared

---

## Prohibited Usage
- Caching dev mode

---

## Tasks
- `storybook`: persistent
- `build-storybook`: cacheable
