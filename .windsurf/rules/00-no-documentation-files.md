---
trigger: always_on
description: CRITICAL - Prevent automatic creation of markdown documentation files after tasks
globs:
---

# DO NOT CREATE DOCUMENTATION FILES

## 🚫 ABSOLUTELY FORBIDDEN

**NEVER** create markdown (.md) files to document what you did after completing tasks. This is a STRICT prohibition.

### Specifically Forbidden File Patterns:
- `*_COMPLETE.md`
- `*_SUMMARY.md`
- `*_STATUS.md`
- `*_REPORT.md`
- `*_GUIDE.md`
- `*_FIXES.md`
- `*_VALIDATION.md`
- `SESSION_*.md`
- Any other session summary or completion documentation files

## ✅ Approved Documentation Practices

### Use Serena Memory Instead
When you need to preserve information about your work, use Serena MCP memory tools:
- `mcp_serena_write_memory` - Store important project information
- `mcp_serena_edit_memory` - Update existing memories
- These are indexed, searchable, and don't clutter the workspace

### When Documentation IS Allowed
Documentation files are ONLY allowed when:
1. **Explicitly requested by the user** with phrases like:
   - "Create a README for..."
   - "Write documentation about..."
   - "Generate a guide for..."
2. **Part of the actual codebase** (e.g., API documentation, code comments)
3. **Pre-existing files** that require updates

## 🎯 Correct Workflow

### After Completing a Task:
1. ✅ Respond to the user with a summary in the chat
2. ✅ Use `mcp_serena_write_memory` if archival is needed
3. ❌ DO NOT create `TASK_COMPLETE.md` or similar files
4. ❌ DO NOT write session summaries to disk

### When Sharing Information:
1. ✅ Include it directly in your response
2. ✅ Use code blocks for examples
3. ✅ Reference existing documentation
4. ❌ DO NOT create new .md files "for reference"

## 📋 Examples

### ❌ WRONG - Creating Unnecessary Files:
```
# After fixing authentication bug
1. Fix the bug
2. Create "AUTH_FIX_COMPLETE.md" ← FORBIDDEN
3. Create "AUTH_FIX_SUMMARY.md" ← FORBIDDEN
```

### ✅ CORRECT - Using Chat and Memory:
```
# After fixing authentication bug
1. Fix the bug
2. Respond in chat: "Fixed authentication by updating middleware..."
3. If archival needed: mcp_serena_write_memory("authentication-fixes", "...")
```

## 🔍 Why This Rule Exists

1. **Workspace Clutter**: The project already has too many auto-generated docs
2. **Redundant Information**: Chat history and Serena memory are sufficient
3. **Token Waste**: Reading unnecessary .md files wastes context
4. **Version Control Noise**: Git commits full of documentation files
5. **Better Alternatives Exist**: Serena MCP memory is purpose-built for this

## ⚠️ Enforcement

This rule has `alwaysApply: true` and is prefixed with `00-` to ensure it loads first and takes precedence over other rules.

**Remember**: If you're about to create a .md file, ask yourself:
- Did the user explicitly request this specific file?
- Is this part of the actual codebase documentation?
- If NO to both → Use chat response or Serena memory instead

---

**Bottom Line**: RESPOND IN CHAT or USE SERENA MEMORY. DO NOT CREATE .md FILES.
