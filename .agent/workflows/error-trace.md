---
description: Expert error tracking and observability specialist focused on implementing comprehensive monitoring, alerting, and structured logging solutions.
---

# Error Tracking and Monitoring

## 1. Strategy & Codebase Analysis

### Observability Lifecycle
- **Detection**: Real-time capture of crashes and unhandled exceptions via specialized SDKs (Sentry, Honeycomb, Datadog).
- **Enrichment**: Attach internal state (user ID, breadcrumbs, memory usage, environment) to every error event.
- **Correlation**: Link errors to specific deployments, request IDs, and infrastructure metrics using OpenTelemetry.
- **Monitoring**: Visualize error rates, latency (P99/P95), and throughput to detect "silent" regressions before users report them.

### Audit Checklist
Analyze your codebase for these critical monitoring gaps:
- **Generic Catches**: Search for `catch (e) {}` or `except Exception: pass` which hide root causes and prevent reporting.
- **Unhandled Promises**: Use linters (e.g., `eslint-plugin-promise`) to ensure every async flow has an error handler.
- **Missing Context**: Identify logs that report "Error: Task failed" without correlation IDs or stack traces.
- **PII Leaks**: Audit log statements that might accidentally record passwords, tokens, or PII (e.g., email addresses in URLs).

---

## 2. Service Integration

### Sentry Setup (Node.js/TypeScript)
Sentry provides a mature ecosystem for grouping and alerting. Configure it with strict PII scrubbing and performance sampling.
```javascript
import * as Sentry from "@sentry/node";

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  release: `v${process.env.APP_VERSION}`,
  
  // Performance Monitoring: 10% sampling for high volumes
  tracesSampleRate: 0.1,
  
  integrations: [
    new Sentry.Integrations.Http({ tracing: true }),
    new Sentry.Integrations.Express({ app }),
  ],
  
  beforeSend(event, hint) {
    // 1. Scrub Sensitive Headers
    if (event.request?.headers) {
      delete event.request.headers['authorization'];
      delete event.request.headers['cookie'];
    }
    
    // 2. Custom Fingerprinting
    // Group errors by type/code rather than variable messages
    const error = hint.originalException;
    if (error?.code === 'ECONNRESET' || error?.name === 'DatabaseError') {
      event.fingerprint = ['database-connection-issue'];
    }
    
    return event;
  }
});
```

### Global Process Handlers
Never let the process die silently. Always catch the ultimate failures for logging and reporting.
```javascript
process.on('uncaughtException', (err) => {
  Sentry.captureException(err);
  console.error('[FATAL] Uncaught Exception:', err);
  // Optional: Give logger/Sentry time to flush before exit
  setTimeout(() => process.exit(1), 2000); 
});

process.on('unhandledRejection', (reason) => {
  Sentry.captureException(reason);
  console.error('[ERROR] Unhandled Rejection:', reason);
});
```

---

## 3. Structured Logging

### Logger Implementation (Winston + Elasticsearch)
Structured logs (JSON) are essential for machine-readability in aggregate tools like ELK, Loki, or Datadog.
```typescript
import winston from 'winston';

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }), 
    winston.format.json()
  ),
  defaultMeta: { service: 'order-api', version: '1.2.0' },
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'error.log', level: 'error' })
  ]
});

// Usage with Context
logger.error('Payment processing failed', { 
  orderId: 'ORD-123', 
  userId: 'user-456',
  duration_ms: 450,
  error: someCaughtError 
});
```

---

## 4. Alerting Configuration

### Threshold-Based Rules
Avoid alert fatigue by setting thresholds on windows. A single error is a blip; a 5% failure rate is an incident.
| Alert Name | Condition | Severity | Channels |
| :--- | :--- | :--- | :--- |
| **Error Spike** | `error_rate > 5%` over 5m | Critical | PagerDuty, Slack |
| **P99 Latency** | `latency > 2s` over 10m | Warning | Slack |
| **OOM Risk** | `memory_usage > 90%` | High | Slack, Email |
| **Silent Fail** | `successful_requests == 0` | Critical | PagerDuty, SMS |

### Logic (Python Concept)
```python
def evaluate_alert(metric_name, current_val):
    rule = alert_registry.get(metric_name)
    if current_val > rule.threshold:
        if not is_silenced(rule) and not in_cooldown(rule):
            dispatch_alert(rule, current_val)

def dispatch_alert(rule, val):
    payload = {
        "text": f"🚨 {rule.name} triggered: {val} (Threshold: {rule.threshold})",
        "env": "production",
        "service": "api-gateway"
    }
    # Send to Slack/PagerDuty Webhook
    requests.post(rule.webhook_url, json=payload)
```

---

## 5. Error Grouping & Deduplication

### Fingerprinting Strategy
To prevent thousands of "unique" events for the same bug, group them using stable elements.
1. **Sanitization**: Remove dynamic data (UUIDs, timestamps, session IDs) from the error message.
2. **Key Frames**: Use a hash of the top 3-5 application-level stack frames (ignoring line numbers if they vary frequently).
3. **Hierarchy**: Use error `Code` or `Name` as a primary grouping key.

```python
import hashlib, re

def generate_fingerprint(error_msg, stack_trace):
    # 1. Normalize: replace UUIDs and numbers
    normalized = re.sub(r'[0-9a-fA-F-]{36}', '<uuid>', error_msg)
    normalized = re.sub(r'\d+', '<num>', normalized)
    
    # 2. Extract stack context (function names in app code)
    app_frames = [f for f in stack_trace if is_app_code(f)][:3]
    
    # 3. Hash final string
    key = f"{normalized}|{'|'.join(app_frames)}"
    return hashlib.sha1(key.encode()).hexdigest()
```

---

## 6. Observability Best Practices
- **Correlation IDs**: Inject `X-Correlation-ID` headers at the edge to link logs and errors across distributed services.
- **Trace Sampling**: Capture 100% of errors but only a percentage of successful spans to manage storage costs.
- **Breadcrumbs**: Track UI interactions (clicks, navigation) and log levels *before* an error occurs.
- **Source Maps**: Automatically upload source maps to Sentry/Datadog during the CI build process.
- **Runbooks**: Attach a documentation link to every alert to reduce MTTR (Mean Time To Resolution).
- **Resource Probes**: Use `/healthz` and `/readyz` endpoints to monitor service availability.