VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
QDRANT_CONTAINER_NAME := qdrant_pipeline
QDRANT_IMAGE := qdrant/qdrant:gpu-nvidia-latest

.PHONY: help install setup-qdrant start-qdrant stop-qdrant logs-qdrant ingest search evaluate-quantization clean

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install                 - Run the full installation and setup process."
	@echo "  setup-qdrant            - Create local directories for Qdrant data and config."
	@echo "  start-qdrant            - Start the Qdrant Docker container with GPU support."
	@echo "  stop-qdrant             - Stop and remove the Qdrant Docker container."
	@echo "  logs-qdrant             - View the logs from the Qdrant container."
	@echo "  ingest                  - Ingest documents from a source directory. Usage: make ingest src=./my_docs"
	@echo "  search                  - Perform a hybrid search. Usage: make search query=\"my search query\""
	@echo "  evaluate-quantization   - Benchmark quantization recall. Usage: make evaluate-quantization src=./sample_docs"
	@echo "  clean                   - Remove virtual environment and generated files."

# --- Core Targets ---

install:
	@bash ./install.sh

setup-qdrant:
	@echo "Creating local directories for Qdrant data and config..."
	@mkdir -p qdrant_data
	@mkdir -p qdrant_config
	@printf "storage:\n  performance:\n    async_scorer: true\n" > qdrant_config/config.yaml
	@echo "Qdrant directories and config created."

start-qdrant: setup-qdrant
	@echo "Starting Qdrant Docker container with GPU support..."
	@docker run -d --name $(QDRANT_CONTAINER_NAME) --gpus all \
	  -p 6333:6333 -p 6334:6334 \
	  -v $(CURDIR)/qdrant_data:/qdrant/storage \
	  -v $(CURDIR)/qdrant_config:/qdrant/config \
	  -e QDRANT__GPU__INDEXING=1 \
	  $(QDRANT_IMAGE)
	@echo "Qdrant container is starting. Use 'make logs-qdrant' to monitor."

stop-qdrant:
	@echo "Stopping and removing Qdrant Docker container..."
	@docker stop $(QDRANT_CONTAINER_NAME) || true
	@docker rm $(QDRANT_CONTAINER_NAME) || true
	@echo "Qdrant container stopped and removed."

logs-qdrant:
	@docker logs -f $(QDRANT_CONTAINER_NAME)

ingest:
ifndef src
	$(error "Usage: make ingest src=<path_to_source_directory>")
endif
	@source activate.sh && \
	$(PYTHON) qdrant_pipeline.py ingest --source $(src)

search:
ifndef query
	$(error "Usage: make search query=\"<your_search_query>\"")
endif
	@source activate.sh && \
	$(PYTHON) qdrant_pipeline.py search --query "$(query)"

evaluate-quantization:
ifndef src
	$(error "Usage: make evaluate-quantization src=<path_to_sample_directory>")
endif
	@echo "Preparing for quantization evaluation..."
	@echo '[{"query": "example search query 1"}, {"query": "example search query 2"}]' > queries.json
	@echo "Created sample 'queries.json'. Please edit it with relevant queries for your data."
	@read -p "Press Enter to continue with the evaluation..."
	@source activate.sh && \
	$(PYTHON) qdrant_pipeline.py evaluate-quantization --source $(src) --queries-file queries.json

clean:
	@echo "Cleaning up project..."
	@rm -rf $(VENV_DIR)
	@rm -f .env
	@rm -f qdrant_pipeline.log
	@rm -f queries.json
	@echo "Cleanup complete. Run 'make stop-qdrant' to stop the database."
