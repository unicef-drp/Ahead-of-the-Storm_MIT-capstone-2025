ENV_NAME=aots_env

.PHONY: env

env:
	conda create -y -n $(ENV_NAME) python=3.10
	conda activate $(ENV_NAME) && pip install -r requirements.txt 