# UGRO HPO Optimizations (2026-01-20)

Optimizations for `LoRAFinetuningObjective` in `src/ugro/hpo/objective.py`:

1.  **Memory**: Explicitly `del model`, `gc.collect()`, and `torch.cuda.empty_cache()` after each trial to prevent OOM in Ray usage.
2.  **Data**: Always `shuffle(seed=42)` before `select()` to avoid data bias.
3.  **Config**: `target_modules` passed via params for flexible sweeps.
