#!/bin/bash
# run 'gcloud auth configure-docker' if unauthorized error message
PROJECT="my-google-project"

DOCKER_IMAGE_NAME='serenade-rust-optimized:master'
docker build -t eu.gcr.io/${PROJECT}/${DOCKER_IMAGE_NAME} -f Dockerfile .
docker push eu.gcr.io/${PROJECT}/${DOCKER_IMAGE_NAME}
