.PHONY: help repl
.DEFAULT_GOAL := help

PREFIX := nlp
APP_NAME := search-term-clustering

help: ## Show this help
	@echo "${PREFIX}/${APP_NAME}"
	@echo
	@fgrep -h "##" $(MAKEFILE_LIST) | \
	fgrep -v fgrep | sed -e 's/## */##/' | column -t -s##

repl: ## Run the sentence embedding REPL
repl:
	python src/cluster/repl.py

##
