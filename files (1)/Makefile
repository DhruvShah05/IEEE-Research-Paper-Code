# ============================================================
#  Makefile — LOB Paper Experiment Runner
#
#  Usage:
#    make install_gpu    — install PyTorch (>=2.7, CUDA 12.8 — required for RTX 50xx), then deps
#    make tune           — Optuna-tune XGBoost + RF on both markets ONCE, print YAML to paste
#    make run_baselines  — chance-level / linear baselines only
#    make gpu_info       — check detected GPU and AMP/BF16 support
#    make prepare        — download and pre-process both datasets
#    make test           — run the pytest test suite
#    make smoke          — quick end-to-end smoke test (1 seed, 1 epoch)
#    make run            — full 5-seed batch run (all 11 configs)
#    make run_fi2010     — FI-2010 only
#    make run_crypto     — Crypto only
#    make ablation       — structured-transformer ablation study
#    make threshold      — threshold sensitivity sweep
#    make backtest       — run backtest on all existing run outputs
#    make significance   — run significance tests on existing outputs
#    make plots          — generate all paper figures
#    make clean          — remove generated per-run .npy / .pt artifacts
#    make clean_models   — remove model.pt checkpoints only
#  ============================================================

PYTHON  := python3
PYTEST  := pytest
RUN_ALL := $(PYTHON) run_all.py

# ---------------------------------------------------------------------------
# GPU Setup — RTX 5080 (Blackwell, compute capability sm_120)
# sm_120 is only supported by PyTorch >= 2.7 built against CUDA 12.8.
# torch is deliberately NOT in requirements.txt so this pin is never overwritten.
# ---------------------------------------------------------------------------

.PHONY: install_gpu
install_gpu:
	@echo "=== Installing PyTorch (>=2.7, cu128) for RTX 50-series ==="
	pip install "torch>=2.7" torchvision torchaudio \
	    --index-url https://download.pytorch.org/whl/cu128
	@echo "=== Installing remaining dependencies ==="
	pip install -r requirements.txt
	@echo "=== Verifying that the GPU is actually usable (not just detected) ==="
	$(PYTHON) -c "import torch; x=torch.randn(64,64,device='cuda'); print('CUDA matmul OK:', (x@x).shape)"
	@echo "=== Verifying GPU ==="
	$(PYTHON) -c "\
import torch; \
print(f'PyTorch  : {torch.__version__}'); \
print(f'CUDA     : {torch.version.cuda}'); \
avail = torch.cuda.is_available(); \
print(f'GPU avail: {avail}'); \
[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB') for i in range(torch.cuda.device_count())] if avail else None; \
print(f'BF16     : {torch.cuda.is_bf16_supported() if avail else False}')"

.PHONY: gpu_info
gpu_info:
	$(PYTHON) -c "\
import torch; \
print(f'PyTorch  : {torch.__version__}'); \
print(f'CUDA     : {torch.version.cuda}'); \
avail = torch.cuda.is_available(); \
print(f'GPU avail: {avail}'); \
[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB') for i in range(torch.cuda.device_count())] if avail else None; \
print(f'BF16     : {torch.cuda.is_bf16_supported() if avail else False}'); \
print(f'TF32     : {torch.backends.cuda.matmul.allow_tf32 if avail else \"N/A\"}')"

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

.PHONY: prepare
prepare:
	@echo "=== Preparing FI-2010 ==="
	$(PYTHON) scripts/prepare_fi2010.py
	@echo "=== Preparing Crypto ==="
	$(PYTHON) scripts/prepare_crypto.py

.PHONY: prepare_fi2010
prepare_fi2010:
	$(PYTHON) scripts/prepare_fi2010.py

.PHONY: prepare_crypto
prepare_crypto:
	$(PYTHON) scripts/prepare_crypto.py

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test
test:
	$(PYTEST) tests/ -v --tb=short

.PHONY: test_fast
test_fast:
	$(PYTEST) tests/ -v --tb=short -x -q

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

.PHONY: smoke
smoke:
	$(RUN_ALL) --smoke-test --seeds 0

# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------

.PHONY: run
run:
	$(RUN_ALL)

.PHONY: run_fi2010
run_fi2010:
	$(RUN_ALL) --markets fi2010

.PHONY: run_crypto
run_crypto:
	$(RUN_ALL) --markets crypto

.PHONY: run_models
run_models:
	$(RUN_ALL) --models $(MODELS)

.PHONY: run_baselines
run_baselines:
	$(RUN_ALL) --models baseline

.PHONY: run_trees
run_trees:
	$(RUN_ALL) --models xgboost random_forest

.PHONY: run_neural
run_neural:
	$(RUN_ALL) --models deeplob transformer structured_transformer

# ---------------------------------------------------------------------------
# Sub-experiments
# ---------------------------------------------------------------------------

.PHONY: ablation
ablation:
	$(PYTHON) experiments/ablation_structured_transformer.py

.PHONY: threshold
threshold:
	$(PYTHON) experiments/threshold_sweep.py

.PHONY: backtest
backtest:
	$(PYTHON) experiments/backtest.py

.PHONY: multi_asset
multi_asset:
	$(PYTHON) experiments/multi_asset.py

.PHONY: horizon_sweep
horizon_sweep:
	$(PYTHON) experiments/horizon_sweep.py

.PHONY: microstructure
microstructure:
	$(PYTHON) experiments/microstructure_stats.py

.PHONY: feature_set
feature_set:
	$(PYTHON) experiments/feature_set_comparison.py

.PHONY: imbalance_ablation
imbalance_ablation:
	$(PYTHON) experiments/imbalance_ablation.py

# ---------------------------------------------------------------------------
# HP Tuning (fix 0.5 / A3 — run ONCE, then paste the printed params into the YAMLs)
# ---------------------------------------------------------------------------

.PHONY: tune
tune:
	$(PYTHON) scripts/tune_all.py --n-trials $(or $(TRIALS),30)

.PHONY: tune_xgboost
tune_xgboost:
	$(PYTHON) scripts/tune_all.py --models xgboost --n-trials $(or $(TRIALS),30)

.PHONY: tune_rf
tune_rf:
	$(PYTHON) scripts/tune_all.py --models random_forest --n-trials $(or $(TRIALS),30)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

.PHONY: significance
significance:
	$(PYTHON) eval/run_significance.py

.PHONY: plots
plots:
	$(PYTHON) eval/plots.py

.PHONY: aggregate
aggregate:
	$(PYTHON) -c "from eval.aggregate import aggregate_results; aggregate_results()"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

.PHONY: clean
clean:
	find results/ -name "*.npy" -delete
	find results/ -name "model.pt" -delete
	find results/ -name "config_used.json" -delete
	find results/ -name "training_history.json" -delete
	@echo "Cleaned run artifacts."

.PHONY: clean_models
clean_models:
	find results/ -name "model.pt" -delete
	@echo "Cleaned model checkpoints."

.PHONY: clean_all
clean_all: clean
	rm -rf results/
	@echo "Removed all results."
