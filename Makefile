.PHONY: create-env install-deps clean 

ENV_NAME=aots_env
PYTHON_VERSION=3.10

# Create the conda environment
create-env:
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

# Install dependencies into the environment
install-deps:
	conda run -n $(ENV_NAME) pip install -r requirements.txt

# Remove the environment (for cleanup)
clean:
	conda env remove -n $(ENV_NAME) -y 