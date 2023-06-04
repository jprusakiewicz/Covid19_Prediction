.PHONY: download-bash


URL := $(shell cat data/data_url.txt)
download-bash:
	wget -O data/covid_data.json $(URL)

run_mlflow:
	./start_mlflow.sh

kill_mlflow:
	./kill_mlflow.sh