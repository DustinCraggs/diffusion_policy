TAG=dp_dc

build:
	docker build ${DOCKER_OPTS} --tag ${TAG} .

run:
	docker run ${DOCKER_OPTS} \
	-it \
	--gpus all \
	-e WANDB_API_KEY=${WANDB_API_KEY} \
	-v $(shell pwd):/diffusion_policy \
	--shm-size=10.06gb \
	${TAG}
