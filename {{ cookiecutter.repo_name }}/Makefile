PHONY: install_poetry clean install_dependencies activate mlflow_server flow bento_save bento_build service_start bento_deploy
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = "{{ cookiecutter.project_name }}"

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install env manager
install_poetry:
	curl -sSL https://install.python-poetry.org | python3 -
# For more details see : https://python-poetry.org/docs/#installation

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Install python dependencies
install_dependencies:
	poetry install

## Activate python interpreter environment
activate:
	poetry shell

## Launch a mlflow server
mlflow_server:
	mkdir -p .mlflow
	mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination ./.mlflow/mlartifacts

## Run experiment flow with prefect
flow:
	find data reports -name ".gitkeep" -type f -delete
	python -m experiment_flow.flow

## Save model to bento format
bento_save:
	python bento/save_model.py

## Build bento then build docker image
bento_build:
	cd ./bento/service; bentoml build -f bentofile.yaml
	bentoml containerize pmb-model:latest -t pmb-model:latest

## Run service on docker
service_start:
	docker run -it --rm -p 3000:3000 pmb-model:latest serve

## Equivalent to bento_save + bento_build + service_start
bento_deploy: bento_save bento_build service_start

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')