# Agent Guide

## Repo Working Agreement

- Use `uv` for environment and package management.
- Prefer `uv sync` to create/update the environment from `pyproject.toml`.
- Prefer `uv run ...` for commands, for example `uv run python meta_main.py qm9_regr --target mu`.
- If we add dependencies, do it in a branch and include the dependency change in the same PR as the code that needs it.

## Git Workflow

- Do not develop directly on `main`.
- Create a branch for every task, open a PR, and merge only after review.
- Do not force-push `main`.
- Do not rewrite history on shared branches unless the whole team explicitly agrees.
- Keep PRs small enough to review: one workstream per PR where possible.
- Before opening a PR, run the smallest relevant smoke test locally and record the command in the PR description.

## Main Branch Safety

- Treat `main` as deployment-sensitive because it is also used to launch scripts on the supercomputer over SSH.
- Do not edit or run ad hoc experiments from `main`.
- Pull `main`, branch off it, and do all work in feature branches.
- Merge to `main` only changes that are reviewed, reproducible, and safe for shared cluster usage.
- Be especially careful with training configs, SLURM scripts, checkpoint paths, and dataset paths because these can break running remote jobs.

## Review Expectations

- Every code PR should be checked by at least one teammate who did not author it.
- Model changes should include:
  - the exact command used for a smoke test,
  - expected config changes,
  - any backward-compatibility risk for existing runs.
- Infra or config PRs should state whether they affect Snellius paths, job scripts, or existing checkpoints.
