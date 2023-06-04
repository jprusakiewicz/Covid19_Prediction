#!/usr/bin/env bash

lsof -i tcp:"${1:-5000}" | awk 'NR!=1 {print $2}' | xargs kill