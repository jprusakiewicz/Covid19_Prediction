.PHONY: download-bash


URL := $(shell cat data/data_url.txt)
download-bash:
	wget -O data/covid_data.json $(URL)

run_mlflow:
	mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root ./mlruns
