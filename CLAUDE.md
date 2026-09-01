# Project: frankenrl

Consolidated, installable framework for composable hybrid RL agents (SAC/TD3/PPO parts + swappable advantage estimators) — the Frankenstein FYP line, unified. Local dev + NCI Gadi PBS runs.

A code project in the **Garage** workspace (`C:\Users\deanc\Garage\`). This file is the
project-specific brief; house rules, the shared agents/commands/hooks, and model routing
live in `~/.claude/CLAUDE.md` and load automatically.

## Scope & goal

- **Goal:** Consolidated, installable framework for composable hybrid RL agents (SAC/TD3/PPO parts + swappable advantage estimators) — the Frankenstein FYP line, unified. Local dev + NCI Gadi PBS runs.
- **In scope:** _fill in_
- **Out of scope:** _fill in_
- **Status (2026-09-01):** scaffolded; stack not chosen yet.

## Stack

_Undecided â€” record the choice here once made (language, framework, package manager,
test runner)._

## Layout

- `src/` â€” source
- `tests/` â€” tests
- (adjust once the stack is picked)

## Garage tools

This project runs inside Garage, so its tooling is available:

- **Grunt lane** â€” `..\..\grunt.ps1 "reformat / commit message / boilerplate â€¦"` sends the
  task to free Gemini Flash (~2 s) instead of spending Claude Code tokens. Piped stdin works:
  `Get-Content x -Raw | ..\..\grunt.ps1 "â€¦"`. The `~/.claude` routing rule already nudges
  grunt-tier subtasks here.
- **Jarvis assistant** â€” `..\..\jask.ps1 "question"` for a fast local answer;
  `..\..\jarvis.ps1 -a orchestrator` for an interactive session with shell/git/file tools
  (confirms before each action).
- **Knowledge vault** â€” `..\..\DeanVault\` (start at `INDEX.md`). Read it for grounding.
  Durable, reusable knowledge you learn while building this â†’ goes **in the vault** per its
  style guide, not in this repo. Code stays here.
- **Shared toolkit** â€” `~/.claude` agents (`planner`, `architect`, `tdd-guide`,
  `code-reviewer`, `security-reviewer`, â€¦) and commands (`/plan`, `/tdd`, `/code-review`,
  `/verify`, â€¦) work here with no setup.

## Conventions

Follows the global rules in `~/.claude/rules/` â€” immutability, small files, comprehensive
error handling, input validation, TDD with 80%+ coverage, conventional-commits, security
checklist before every commit.

