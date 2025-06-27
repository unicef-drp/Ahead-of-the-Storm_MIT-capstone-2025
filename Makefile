.PHONY: env update-env

env:
	conda create -y -n $(ENV_NAME) python=3.10
	conda activate $(ENV_NAME) && pip install -r requirements.txt

update-env:
	conda activate $(ENV_NAME) && pip install -r requirements.txt 