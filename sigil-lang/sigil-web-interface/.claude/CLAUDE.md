# ZSH Aliases

```bash
pf               # cd to persona-framework root
lev-clean        # Clean mammon + leviathan builds
lev-build        # Build leviathan (-x test)
lev-run          # Export AWS creds + bootRun
lev-kill         # Kill all leviathan processes
lev-restart      # Full restart: kill + clean + build + run
lev-logs         # tail -f leviathan.out
bael-dev         # cd bael && npm run dev
bael-test        # cd bael && playwright headed mode
eidolon          # Launch script (all|backend|gui|stop|restart|status)
eidolon-start    # Start backend + GUI
eidolon-stop     # Stop all components
eidolon-restart  # Restart all components
eidolon-status   # Show component status
eidolon-backend  # Start backend only
eidolon-gui      # Start GUI only
```

# Grimoire Integration

Prompts MUST be loaded from `/home/lilith/development/projects/grimoire/personas/` at runtime.
PromptLoader configured to read from filesystem, NOT packaged JARs.

# Git Workflow

**CRITICAL**: Never work directly on main or development branches.

For each roadmap/goal:
1. Create feature branch: `git checkout -b feature/goal-name`
2. Commit regularly as you make progress
3. When complete, create PR for review

Example:
```bash
git checkout -b feature/streaming-chat
# make changes
git add -A && git commit -m "feat: streaming implementation"
# continue work...
git push origin feature/streaming-chat
```

# Common Workflows

Build + Run Leviathan:
```bash
pf && lev-clean && lev-build && lev-run
```

Run E2E Test:
```bash
pf && cd bael && npx playwright test hydra-marketing-task.e2e.spec.ts:18 --headed --project=chromium
```
