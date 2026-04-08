# Wiki Schema

This wiki is an LLM-maintained knowledge base for Cython optimization patterns.
It sits between raw sources (Cython docs, our cy/ implementations, traces) and
the agents that write Cython code.

## Raw Sources (read-only, never modified by the wiki)

- `.sources/cython-docs/` — official Cython documentation (gitignored)
- `cnake_data/cy/` — our 500+ Cython implementations across 19 categories
- `data/traces/master_traces.jsonl` — 10K+ agent traces with error/fix patterns

## Page Template

Every page in `pages/` should follow this structure:

```markdown
# Page Title

One-line summary of what this page covers.

## Overview

Brief explanation of the concept and when to use it.

## Syntax

Core syntax examples with minimal explanation.

## Patterns

Real-world patterns extracted from our codebase and traces.
Each pattern should include:
- The Cython code
- Why it's fast (what C-level optimization it enables)
- Source attribution (cy/ file or trace if applicable)

## Gotchas

Common mistakes observed in traces. Each gotcha should include:
- What goes wrong
- How it manifests (compilation error, segfault, wrong output)
- The fix

## See Also

Cross-references to related pages: [[page-name]]
```

## Conventions

- **Filenames**: kebab-case, `.md` extension (e.g., `memory-management.md`)
- **Cross-references**: `[[page-name]]` or `[[page-name#section]]`
- **Code blocks**: Always use `cython` language tag for Cython code
- **Source attribution**: Note where content came from (`Source: cy/numerical/great_circle.pyx` or `Source: traces, 47 occurrences`)
- **No duplication**: If content belongs on another page, link to it instead
- **Examples over prose**: Show the code, explain briefly why it works

## Operations

- **Ingest**: Read a raw source, extract insights, update relevant pages
- **Reflect**: Analyze traces for patterns, update Gotchas and Patterns sections
- **Lint**: Check for orphaned pages, broken cross-refs, stale content

## Special Files

- `index.md` — content catalog, updated whenever pages are added/modified
- `log.md` — append-only record of all wiki operations
- `schema.md` — this file, the rules of the wiki
