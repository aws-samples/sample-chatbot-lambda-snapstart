build-LlamaCppLayer:
	docker build -t llama-cpp-layer .
	docker create --name temp_container llama-cpp-layer
	docker cp temp_container:/opt/. $(ARTIFACTS_DIR)/
	docker rm temp_container

