PYTHON ?= python

STAGE1_SCRIPT := compute_feature_importances.py
STAGE2_SCRIPT := train_stage2.py

.PHONY: all stage1 stage2 clean

all: stage1 stage2

stage1:
	$(PYTHON) $(STAGE1_SCRIPT)

stage2:
	$(PYTHON) $(STAGE2_SCRIPT)

clean:
	rm -f model_cv_results_parallel.csv model_cv_results_parallel.json
	rm -rf trained_models
