# Hatırlaf — developer entrypoint.
#
#   make            list every target
#   make setup      create .venv and install everything
#   make run        start the server at http://127.0.0.1:8000
#   make test       run the backend test suite
#
# Every target is a thin wrapper over the scripts in scripts/ or over
# manage.py, so nothing here hides behaviour you cannot reproduce by hand.

SHELL   := /usr/bin/env bash
ROOT    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV    := $(ROOT)/.venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
SERVER  := $(ROOT)/server
MOBILE  := $(ROOT)/clients/mobile
MANAGE  := cd $(SERVER) && $(PYTHON) manage.py

HOST ?= 127.0.0.1
PORT ?= 8000

.DEFAULT_GOAL := help

# --- Getting started ---------------------------------------------------------

## help: list the available targets
help:
	@echo "Hatırlaf"
	@echo
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  make /' | column -t -s ':'
	@echo
	@echo "  The NLP layer is off by default. Turn it on for one run with:"
	@echo "    make run NLP=1"

## setup: create the virtualenv and install all dependencies (safe to re-run)
setup:
	@scripts/setup.sh

## setup-minimal: setup without the multi-GB ML stack (no STT, NER, or LLM)
setup-minimal:
	@HATIRLAF_SETUP_MINIMAL=1 scripts/setup.sh

## run: migrate, then serve the app (override with HOST=, PORT=, NLP=1)
run:
	@HATIRLAF_HOST=$(HOST) HATIRLAF_PORT=$(PORT) \
	 $(if $(NLP),HATIRLAF_NLP_ENABLED=$(NLP),) scripts/run.sh

# --- Everyday development ----------------------------------------------------

## test: run the backend test suite
test:
	@$(MANAGE) test diary

## migrate: apply database migrations
migrate:
	@$(MANAGE) migrate

## migrations: generate migrations for the diary app
migrations:
	@$(MANAGE) makemigrations diary

## shell: open a Django shell with the app loaded
shell:
	@HATIRLAF_PRELOAD_MODELS=0 $(MANAGE) shell

## seed: fill the local database with demo entries
seed:
	@$(MANAGE) seed_demo

## admin: create a Django superuser for /admin/
admin:
	@$(MANAGE) createsuperuser

# --- Clients -----------------------------------------------------------------

## mobile: start the Expo dev server for the mobile client
mobile:
	@cd $(MOBILE) && npm install && npm start

# --- Housekeeping ------------------------------------------------------------

## clean: delete caches and collected static files (keeps db and media)
clean:
	@find $(ROOT) -name __pycache__ -type d -not -path '*/.venv/*' -prune -exec rm -rf {} +
	@rm -rf $(ROOT)/var/staticfiles
	@echo "Cleaned caches. var/db.sqlite3 and var/media/ were left alone."

## reset: delete the local database and all recorded media — destructive
reset:
	@read -p "Delete var/db.sqlite3 and everything in var/media/? [y/N] " ok && \
	 [ "$$ok" = "y" ] && rm -rf $(ROOT)/var/db.sqlite3 $(ROOT)/var/media && \
	 echo "Reset. Run 'make migrate' to start fresh." || echo "Aborted."

.PHONY: help setup setup-minimal run test migrate migrations shell seed admin mobile clean reset
