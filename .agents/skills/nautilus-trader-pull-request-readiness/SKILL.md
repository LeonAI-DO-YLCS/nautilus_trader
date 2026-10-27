---
name: nautilus-trader-pull-request-readiness
description: >-
  Load when you prepare, update or claim ready a pull request or a review-ready change in the
  nautilus_trader project: the local checks to run (make format, make pre-commit, relevant tests,
  make pre-flight), how to treat validation evidence and CI, and how to handle review feedback.
  Moved word for word from the project AGENTS.md section "Pull request readiness" (2026-10-26).
---

# Pull request readiness (moved word for word from AGENTS.md)

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
