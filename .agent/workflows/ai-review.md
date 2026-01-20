---
description: # AI-Powered Code Review Specialist
---

# AI-Powered Code Review Specialist

Expert code reviewer combining AI tools (Claude 4.5 Sonnet, GPT-5, GitHub Copilot) with static analysis platforms (SonarQube, CodeQL, Semgrep) to identify bugs, vulnerabilities, and performance issues.

## Review Workflow

### 1. Initial Triage
- Parse diff for modified files and components
- Match file types to optimal analysis tools
- Scale depth based on PR size (<200 lines = deep, >1000 = superficial)
- Classify change type: feature, bug fix, refactoring, breaking change

### 2. Multi-Tool Analysis (Parallel)
- **CodeQL**: Vulnerability scanning (SQL injection, XSS, auth bypasses)
- **SonarQube**: Code smells, complexity, duplication
- **Semgrep**: Custom security policies
- **Snyk/Dependabot**: Dependency vulnerabilities
- **GitGuardian**: Secret detection

### 3. AI-Assisted Review

**Model Selection (2025):**
- Fast reviews (<200 lines): GPT-4o-mini, Claude 4.5 Haiku
- Deep reasoning: Claude 4.5 Sonnet, GPT-5
- Code generation: GitHub Copilot, Qodo

**Review Prompt Template:**
```python
review_prompt = f"""
Review this {language} pull request.

**Changes:** {pr_description}
**Code Diff:** {code_diff}
**Static Analysis:** {tool_issues}

Focus on:
1. Security vulnerabilities missed by static tools
2. Performance at scale
3. Edge cases and error handling
4. API compatibility
5. Missing test coverage
6. Architecture alignment

For each issue provide:
- File path and line number
- Severity: CRITICAL/HIGH/MEDIUM/LOW
- Problem explanation (1-2 sentences)
- Fix example
- Reference documentation

Return as JSON array.
"""
```

## Key Analysis Areas

### Architecture
- **SOLID Principles**: Single Responsibility, Open/Closed, Dependency Inversion
- **Anti-patterns**: God objects (>500 lines), Singletons, Anemic models
- **Microservices**: Service cohesion, database-per-service, API versioning

### Security (OWASP Top 10)
1. Broken Access Control (IDOR, missing authorization)
2. Cryptographic Failures (weak hashing, insecure RNG)
3. Injection (SQL, NoSQL, command injection)
4. Insecure Design (missing threat modeling)
5. Security Misconfiguration (default credentials)

### Performance Red Flags
- N+1 queries, missing indexes
- Synchronous external calls
- Unbounded collections, no pagination
- Missing connection pooling
- No rate limiting

```python
def detect_n_plus_1(code_ast):
    for loop in find_loops(code_ast):
        db_calls = find_database_calls(loop.body)
        if db_calls:
            return {
                'severity': 'HIGH',
                'message': f'{len(db_calls)} DB calls in loop',
                'fix': 'Use eager loading (JOIN) or batch loading'
            }
```

## Review Comment Format

```typescript
interface ReviewComment {
  path: string;
  line: number;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  category: 'Security' | 'Performance' | 'Bug' | 'Maintainability';
  title: string;
  description: string;
  codeExample?: string;
  cwe?: string;
  effort: 'trivial' | 'easy' | 'medium' | 'hard';
}
```

**Example:**
```json
{
  "path": "src/auth/login.ts",
  "line": 42,
  "severity": "CRITICAL",
  "category": "Security",
  "title": "SQL Injection Vulnerability",
  "description": "String concatenation enables SQL injection. Attack: 'admin' OR '1'='1'",
  "codeExample": "// ✅ Use: db.execute('SELECT * WHERE user = ?', [username])",
  "cwe": "CWE-89",
  "effort": "easy"
}
```

## CI/CD Integration

```yaml
name: AI Code Review
on: pull_request

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Static Analysis
        run: |
          sonar-scanner
          codeql database create --language=javascript,python
          semgrep scan --config=auto --sarif
      
      - name: AI Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python scripts/ai_review.py \
            --pr ${{ github.event.number }} \
            --model claude-sonnet-4-5
      
      - name: Quality Gate
        run: |
          CRITICAL=$(jq '[.[]|select(.severity=="CRITICAL")]|length' results.json)
          [ $CRITICAL -eq 0 ] || exit 1
```

## Implementation Example

```python
from anthropic import Anthropic

class CodeReviewer:
    def __init__(self, pr_number: int):
        self.pr_number = pr_number
        self.client = Anthropic()
    
    def run_static_analysis(self):
        # Run SonarQube, Semgrep, CodeQL
        return {'semgrep': [], 'sonarqube': []}
    
    def ai_review(self, diff: str, static_results: dict):
        prompt = f"""Review PR: {diff[:10000]}
Static Analysis: {static_results}

Return JSON: [{{"path","line","severity","title","description","fix"}}]"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self.parse_issues(response.content[0].text)
    
    def post_comments(self, issues):
        critical = sum(1 for i in issues if i['severity'] == 'CRITICAL')
        status = 'REQUEST_CHANGES' if critical > 0 else 'APPROVE'
        # Post to GitHub API
```

## Summary

Complete AI code review pipeline:
- Multi-tool static analysis (SonarQube, CodeQL, Semgrep)
- AI-powered deep analysis (Claude 4.5 Sonnet, GPT-5)
- CI/CD integration (GitHub Actions, GitLab)
- Actionable feedback with severity levels and fixes
- Quality gates preventing critical issues
- 30+ language support

Transform manual reviews into automated AI-assisted quality assurance with instant, comprehensive feedback.