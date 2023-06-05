#!/usr/bin/env bash

nohup mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root ./mlruns > /dev/null &