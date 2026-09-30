# V18 prefit import amendment

The first invocation of `training/train_mmbert_factors.py` exited with `ModuleNotFoundError: No module named 'training'` at its import line. Python had placed the `training/` script directory, rather than the repository root, on its script import path. This occurred before `main()`, model loading, output creation, optimizer construction, or access to batch 0034. No fit or inference occurred.

The import was changed from `training.v18_facts` to `v18_facts`, which resolves the adjacent module under the registered script invocation. The dataset, prompts, optimizer, epochs, selection rule, and success target in [V18_PROTOCOL.md](V18_PROTOCOL.md) are unchanged. This amendment and corrected trainer are committed before the rerun. The failed launch remains part of the execution record; it is not counted as a model experiment.
