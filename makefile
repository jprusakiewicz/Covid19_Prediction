.PHONY: download-bash


URL := $(shell cat data/data_url.txt)
download-bash:
	wget -O data/covid_data.json $(URL)

run_mlflow:
	./scripts/start_mlflow.sh

kill_mlflow:
	./scripts/kill_mlflow.sh