# Agent Instructions

Read [AI_POLICY.md](AI_POLICY.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before making changes.
Follow the [coding standards](docs/developer_guide/coding_standards.md) and the relevant developer
guide for the area you change.

## Working rules

**NautilusTrader can execute live trades involving real capital. Hold every change to a very high
standard for correctness, reliability, testing, clarity, and maintainability.**

### Environment

- Use the uv-managed Python project environment at its default location, `python/.venv`. Do not
  create or use a root `.venv`; run direct uv project commands from `python/` or pass
  `--project python` from the repository root.

### Scope and style

- Read the affected code and search for existing patterns before proposing or making changes.
- Keep each change focused on the requested outcome with the minimum public API. Note unrelated
  issues instead of fixing them, and avoid drive-by refactors, renames, and abstractions
  unrelated to the contribution.
- Match the existing style and use established functions, types, names, and dependencies.

### Correctness

- Preserve exact arithmetic for prices, quantities, money, fees, and other discrete values. Use the
  project domain types or `Decimal`.
- Keep scripts and paths portable across Linux, macOS, and Windows. Normalize path separators with
  the shared helpers (for example `normalize_path_separators` in
  `nautilus_persistence::common::paths`) before splitting or joining; never split a raw path on
  `/` alone.

### Tests

- Do not add test-only behavior, branches, attributes, or interfaces to production code.
- Do not weaken, remove, bypass, or rewrite tests or required behavior merely to obtain a passing
  result. Fix the underlying problem and preserve the behavior the tests are intended to protect.
  Change a test only when the task intentionally changes the required behavior or when you can
  independently verify that the test is wrong.
- Do not let tests depend on adapter environment variables; `make pre-flight` runs with all of
  them unset via `scripts/strip-adapter-env.bash`. Register new adapter environment variables there.

### Change mechanics

- Change generated artifacts through their source and generator. Never edit them by hand.
- Do not modify `RELEASES.md`. Maintainers keep it current.
- Do not modify `.github/workflows` or `.github/actions` for an external contribution. These paths
  are maintainer-only.

### Commit messages

- Do not use Conventional Commits syntax for commit messages or pull request titles.
- Do not put an issue or pull request number in a commit subject or pull request title. A squash
  merge appends the number; reference issues from the commit body instead.

## Pull request readiness

**Prepare a complete, review-ready change before opening a pull request.**

Run the smallest relevant test while developing. Before opening or updating a pull request,
establish the development environment described in `CONTRIBUTING.md`. Run `make format`,
`make pre-commit`, and all tests relevant to the change locally, and confirm they pass. If a
required check cannot run, report the limitation and do not claim the change is ready. Follow the
pull request template when preparing the pull request description.

Treat validation as evidence for the exact tested change. After a later edit or rebase, rerun the
affected checks.

For higher assurance, run `make pre-flight`, which performs the project's broad local validation
suite.

Treat project CI as confirmation of a locally validated change, not as a development loop or a
substitute for local compute. If a maintainer asks for an early draft, agree on its scope before
opening it. After review feedback, batch related fixes and rerun the relevant local checks so the
contributor can push one coherent update.

## Git and public interaction

- Base contributor work on `develop` and target `develop` when preparing a pull request.
- Do not commit, amend, push, or change remote state unless the user explicitly asks.
- Do not open, edit, comment on, review, or otherwise interact with GitHub issues or pull requests
  unless the user explicitly asks. The human contributor controls every public interaction and
  remains responsible for the final communication.
- AI may assist with drafting, but the contributor must understand the text and verify its accuracy.
  When drafting, help preserve the contributor's choices and voice rather than replacing them with
  generic prose.

## Disclosure and attribution

- NautilusTrader does not require disclosure of AI assistance.
- The [AI Policy](AI_POLICY.md) does not override any legal, contractual, or license obligation that
  applies to the contributor or submitted material.
- If the contributor chooses to disclose AI assistance in a commit message, keep the wording
  general, such as `Developed with assistance from AI.`
- The project is neutral among AI labs, vendors, models, and tools, so do not name or promote a
  specific one as attribution in commit messages, pull request titles, or pull request descriptions.
- Do not add an AI tool or model as an author, co-author, or contributor.
- Do not add `Co-authored-by:` trailers for AI tools or models.
- Do not add branded footers such as `Generated with ...` to commit messages or pull request text.

## LeonAI-DO fork rules

These rules come from the fork owner (2026-10-26). They apply to every agent working in this fork
and take precedence over the generic contributor guidance above where they differ.

### Repository, place and sync

- `origin` is the owner's fork, `https://github.com/LeonAI-DO-YLCS/nautilus_trader`. Never pull
  from, push to, or open issues or pull requests on the upstream `nautechsystems` repository.
- Develop, build and run in WSL2 Ubuntu, in the clone `/home/leonai-do/dev/nautilus_trader`, with
  the default `python/.venv`. MetaTrader 5 runs on Windows; reach it only through a loopback
  service on `127.0.0.1` (WSL mirrored networking). The Windows copy at
  `C:\LeonAI-DO\dev\TRADING\nautilus_trader` and the Linux copy sync only through GitHub push and
  pull.
- Put every change on a new branch (small change) or in a git worktree under `.worktrees/`
  (several agents in different scopes). Never commit directly on `develop` or `main`; merging into
  `develop` needs the owner's approval.
- Push each commit to `origin` right away (standing owner authorization for work branches on the
  fork). The commit-message and attribution rules above still apply.

### Priorities

- Efficiency, speed, robustness and accuracy come first. Use the native, fastest exact data paths
  (Parquet catalog, Arrow wranglers, streaming backtests).
- Run a prior-art check before building anything; do not rebuild what already exists in this
  repository or in the owner's other projects. Test, validate and verify every step.

### Engines

- Standalone backtests, strategy optimization, genetic parameter search and strategy mining use
  the free open-source `vectorbt`, never NautilusTrader. It must support binary options and keep
  traditional trading. The vectorbt work lives in its own folder in this repository with its own
  Python environment.
- Only strategies proven in vectorbt move to NautilusTrader, for event-driven confirmation and
  then paper or live trading. Define settlement, payout and stake rules once and share them.

### Deriv binary options profile (first release)

- Market: Volatility 75 (`R_75`), strict Rise/Fall, 1-minute duration. `CALL` = Rise, `PUT` = Fall;
  a tie loses. Fixed 95.2% net payout per run.
- Money: every run starts at 10,000 USD. Stake = 0.05% of the available balance (stakes locked in
  open contracts excluded), rounded down to cents, raised to 0.35 USD when below. Up to 5
  contracts open at once.
- Label results from MT5 data "MT5 simulation" until MT5 and Deriv price feeds are proven equal.
- Deriv API docs index: `https://developers.deriv.com/llms.txt`. Do not install the Deriv MCP
  server until the owner asks.

### Subagent failures

- When a subagent fails the same job the same way twice, the coordinating agent stops delegating
  that job and does it itself. Get the failing agent's report first, then follow every other rule
  here (branch or worktree, tests, no merge into `develop` without the owner).

### Safety

- Never place real-money trades, log in to broker accounts, or read credential files unless the
  owner explicitly asks. Keep secrets out of files, logs and output.
- Test data: `C:\LeonAI-DO\dev\TRADING\Datasets\Volatility75Index_yearly_deduped`. Never write to
  the `Datasets` folder.
