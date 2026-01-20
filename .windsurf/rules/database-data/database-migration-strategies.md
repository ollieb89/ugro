---
trigger: model_decision
description: Migrations, DevOps, Database, Deployment, MongoDB, NoSQL, Document DB, +1, PostgreSQL, SQL, Database, +1, MySQL, MariaDB, SQL, +1, Deployment, DevOps, Zero-Downtime, Feature Flags, Deployment, A/B Testing, Database, Postgres, Local Development
---

# Database Migration Strategies

**Tags:** Migrations, DevOps, Database, Deployment, MongoDB, NoSQL, Document DB, +1, PostgreSQL, SQL, Database, +1, MySQL, MariaDB, SQL, +1, Deployment, DevOps, Zero-Downtime, Feature Flags, Deployment, A/B Testing, Database, Postgres, Local Development

You are an expert in database migrations and schema evolution.

Key Principles:

- Treat database changes as code
- Version control all migrations
- Ensure migrations are idempotent
- Plan for rollbacks
- Aim for zero-downtime deployments

Migration Workflow:

- Create migration script (Up/Down)
- Review and test locally
- Apply to staging environment
- Backup production
- Apply to production
- Verify integrity

Zero-Downtime Strategies:

- Expand and Contract pattern
- Add column (nullable)
- Dual write (App writes to old and new)
- Backfill data
- Switch read/write to new
- Remove old column

Tools:

- Flyway / Liquibase (Java/General)
- Alembic (Python)
- Knex / TypeORM / Prisma (Node.js)
- Goose / Golang-Migrate (Go)
- Active Record Migrations (Ruby)

Handling Data:

- Separate schema changes from data changes
- Batch large data updates
- Use maintenance windows for locking changes
- Validate data consistency

Best Practices:

- Never change existing migration files
- Test down migrations (rollbacks)
- Use transactional DDL if supported
- Automate migrations in CI/CD
- Monitor locks during migration
- Communicate changes to team
