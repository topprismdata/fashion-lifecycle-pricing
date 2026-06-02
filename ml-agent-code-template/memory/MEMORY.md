# Memory Index
<!-- Update with one-line entries as memory grows -->

- [principles/16-principles](principles/16-principles.md) — Cross-domain ML principles (start here)
- [principles/local-optimum-trap](principles/local-optimum-trap.md) — When to pivot vs iterate
- [skills/feature-engineering-roi](skills/feature-engineering-roi.md) — When features > hyperparams
- [skills/stacking-ceiling](skills/stacking-ceiling.md) — When to stop stacking
- [skills/cv-strategy](skills/cv-strategy.md) — Group K-Fold vs Stratified
- [skills/external-data-fusion](skills/external-data-fusion.md) — Highest ROI step
- [skills/500-line-rule](skills/500-line-rule.md) — Skill size limit + progressive disclosure
- [feedback_no_recheck_confirmed_dead](feedback_no_recheck_confirmed_dead.md) — Known dead ends (ALWAYS check first)

## Experiment Records

- `experiments/` — Per-experiment files: `exp_<YYYYMMDD>_<short-name>.md`

## Competition Records

- `competitions/` — Per-competition files: `<competition-slug>.md`

## Template Quick Start

```
# Add new principle
Write memory/principles/<name>.md → add to index

# Add new skill
Write memory/skills/<name>.md → add to index

# Add new dead end
Write memory/feedback_no_recheck_confirmed_dead.md → append

# Log experiment
Write memory/experiments/exp_<YYYYMMDD>_<name>.md
```
## Proactive Evolution (2026-06-01)

- [skills/self-critique-checkpoint](../memory/skills/self-critique-checkpoint.md) — P0 mid-task self-critique (5 questions before executing)
- [skills/path-efficiency-analysis](../memory/skills/path-efficiency-analysis.md) — P2 analyze HOW efficiently (Recovery, Repetitiveness, Tool Productivity, Costliest Wrong Path)

## New Hooks (2026-06-01)

- `hooks/self_critique_trigger.sh` — fires on "let me try/train/run" prompts
- `hooks/path_efficiency.sh` — analyzes session log for 4 efficiency metrics
- `hooks/cross_review_trigger.sh` — extended to also fire on train/execute/run (pre-execution simulation)
