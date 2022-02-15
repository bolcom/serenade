#docker run --rm -i --entrypoint /bin/bash -t serenade-rust-optimized
PROJECT="my-google-project"
DOCKER_IMAGE_NAME='serenade-rust-optimized:master'
docker run --rm -i --entrypoint /bin/bash -t eu.gcr.io/${PROJECT}/${DOCKER_IMAGE_NAME}
