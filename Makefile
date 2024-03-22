.DEFAULT_GOAL := default
X_PRED = pd.DataFrame([np.array(["FR",1,1,1,1,1,6,1,1,1,1])], columns=['cntry', \
'gndr', 'sclmeet', 'inprdsc', 'sclact', 'health', 'rlgdgr','dscrgrp',     \
'ctzcntr', 'brncntr', 'happy'])

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y taxifare || :
	@pip install -e .

# run_preprocess:
# 	python -c 'from taxifare.interface.main import preprocess; preprocess()'

# run_train:
# 	python -c 'from taxifare.interface.main import train; train()'

run_pred:
	python -c 'from howhappyineurope.interface.main_local import pred; pred()'

run_api:
	uvicorn howhappyineurope.api.fast:app --reload

# run_evaluate:
# 	python -c 'from taxifare.interface.main import evaluate; evaluate()'

# run_all: run_preprocess run_train run_pred run_evaluate

# run_workflow:
# 	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m taxifare.interface.workflow

# run_api:
# 	uvicorn taxifare.api.fast:app --reload
