# Owner requests, session 2026-10-26, and where each becomes permanent

Source: the owner's own list of 31 requests (pasted 2026-10-26, about 11:05 PM), plus one new rule.
Owner answers on scope (same evening): requests 14/26, 15, 10 and 6 become GLOBAL; hooks are
proposed first and switched on only after owner approval; the MT5 login folder goes into the
project rules with the safety rule; merging rule-file changes into `develop` is approved for this task.

## The owner's requests (verbatim, numbered as the owner sent them)

1. Visual workflow of how this tool works (architecture, parts, interconnections), built by research subagents plus one HTML builder agent, inspired by the communication card.
2. Backtest binary options on deriv.com markets (not Polymarket): add MT5 as provider, connect to the Deriv broker there, get data, backtest, later trade. MT5 is installed and logged in; credentials, if needed, are in `C:\LeonAI-DO\dev\TRADING`. Test datasets are in `C:\LeonAI-DO\dev\TRADING\Datasets`.
3. Start with Volatility 75. Run the interview and assessment; subagents gather data; then plan MT5 and Deriv as providers and adapt the system to backtest and trade Deriv binary options.
3b. All changes on a new branch if minimal, in a new worktree when many agents work in different scopes.
4. Test with `C:\LeonAI-DO\dev\TRADING\Datasets\Volatility75Index_yearly_deduped` if compatible; else another available one; if none, ask. Every question or decision: short context, pros, cons, recommended option and why.
5. Check what the system accepts and how; efficiency, speed, robustness and accuracy first; use the prior-art gate skill (`.agents/skills/`), do not rebuild what exists; finish as fast as agents can, always testing, validating and verifying.
6. Too much text: answer visually, structured HTML with short context, always.
7. WSL2 for development and running; MT5 runs on Windows.
8. Save all decisions to the logs.
9. Also to the project memory.
10. GPT 6.1 Sol high supervises all work (supervisor, advisor, evaluator, alignment); the main session is manager and coordinator.
11. Save all pending decisions not yet saved.
12. Standalone backtests always use vectorbt; proven results move to NautilusTrader; optimization, genetic parameter search and strategy mining use vectorbt (binary options plus traditional trading), never NautilusTrader.
13. Save important rules to this project's AGENTS.md so every agent has them.
14. Keep working on tasks that do not need the owner; leave questions for the owner's return.
15. After a subagent fails the same way twice, the coordinator does that job itself.
16. Save the owner's requests as project rules.
17. When anything is open or asked again, give short context so the owner understands it.
18. Show the full roadmap with every phase's status in percentages.
19. Model pool: Haiku max, GPT 6 Luna xhigh, GPT 6.1 Sol high, Sonnet 5.5 xhigh; at least 3 agents busy on non-overlapping tasks; pick model and effort by efficiency (Artificial Analysis), check quota; two-fail takeover stays.
20. Save the model research as a file for any project, and in global memory.
21. Global: always check the quota of a model and provider before spawning an agent.
22. Global: after each finished task, show the current phase's roadmap; after a finished phase, the full roadmap with short context, as an HTML card.
23. Global: every roadmap or phase update shows agent time left, based on measured past work.
24. Show the phased roadmap with times and the current phase's tasks with times; add missing global rules.
25. Global: no agent works more than 15 minutes on one task (5 implement, 5 test, 5 validate and report); over time, ask for a report and the reason, then decide.
26. When the owner is away, keep working autonomously on approved tasks that do not wait for another task or phase (research, documenting, organizing, compacting, reporting, standalone future tasks); set timers to keep going.
27. Recalibrate times on every update from measured agent runs plus the coordinator's management time.
28. Use Engram, zvec-grep and CodeGraph on every task, Windows only, never in WSL.
29. Save that globally and in AGENTS.md: all agents use the three tools for memory, retrieval and keeping the project up to date.
30. Reinstall a broken tool if needed, but keep the old database (no data loss). Spawn more agents and be proactive on the roadmap.
31. Global: always show time as agent time (human time).
32. NEW (2026-10-26): always tell every subagent it is a subagent of the parent session; it never works as another entity unless the parent tells it to.

## Classification and target

| # | Kind | Scope | Mandatory | Target |
|---|---|---|---|---|
| 32 | Rule + packet template line | Global | Yes | Global AGENTS.md; `subagent-supervision` skill packet template; every packet |
| 21 | Rule + script | Global | Yes | Exists in global AGENTS.md; add script `C:\Users\yensi\.agents\scripts\codex-quota.mjs` and reference it |
| 22, 23, 24 | Rule | Global | Yes | Exist (global AGENTS.md, `owner-status-reports` skill section 14) |
| 27, 31 | Rule + script | Global | Yes | Exist in global AGENTS.md; add script `C:\Users\yensi\.agents\scripts\agent-times.mjs` (measured task durations) and reference it |
| 25 | Rule | Global | Yes | Exists |
| 28, 29, 30 | Rule | Global + project | Yes | Exist (global AGENTS.md, `integrated-retrieval` skill section 0, project AGENTS.md) |
| 20 | Reference file | Global | Suggestion source | `C:\Users\yensi\.claude\model-efficiency-research.md`; make sure global AGENTS.md or `subagent-models` skill points to it (not only CLAUDE.md) |
| 14, 26 | Rule + workflow | Global (owner choice) | Yes | Global AGENTS.md pointer; full workflow in `subagent-supervision` skill |
| 15 | Rule | Global (owner choice) + project | Yes | Global AGENTS.md; project AGENTS.md keeps it |
| 10 | Rule | Global (owner choice) + project | Yes | Global AGENTS.md and `subagent-models` skill: Sol 6.1 high is the default supervisor/reviewer (replaces "medium by default"); Sonnet 5.5 xhigh backup above about 85% Codex quota; tell the owner; never switch silently |
| 6 | Rule | Global (owner choice) + project | Yes | Global AGENTS.md line: replies are short structured HTML cards with short context |
| 2 | Project rule | Project | Yes | Project AGENTS.md: goal; MT5 on Windows, logged in; login details in `C:\LeonAI-DO\dev\TRADING`, read only when a task needs the MT5 login, never print or copy them into files, logs or output |
| 3, 3b, 7, 12, 19 | Project rule | Project | Yes | Exist in project AGENTS.md (check) |
| 4 | Project rule | Project | Yes | Project AGENTS.md: test dataset, else another compatible one, else ask; question format |
| 5 | Project rule | Project | Yes | Project AGENTS.md: compatibility check first; prior-art gate skill path |
| 8, 9, 11 | Workflow | Project (+ global record rule) | Yes | Project AGENTS.md: every decision goes to the project decision log, Engram (`--project nautilus-trader`) and coordinator memory |
| 13, 16, 17 | Meta / exists | Project | Yes | Project AGENTS.md (exists: "Talking to the owner") |
| 1, 3, 18, 24 | One-off outputs | Project | No | Not rules; record as done tasks in the decision log |
| H1 | Hook (proposed) | Global, Claude Code | Owner approval needed | PreToolUse on `mcp__t3-code__delegate_task`: block a packet that lacks the subagent-identity line or "Owner rules (word for word)" |
| H2 | Hook (proposed) | Global, Claude Code | Owner approval needed | PreToolUse on `mcp__t3-code__delegate_task`: print Codex quota (non-blocking) |
