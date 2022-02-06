notebook:
	docker run --rm --network host --gpus all -p 8888:8888 --name python-env -v ${PWD}:/home/jovyan/work python-env

build:
	docker build -t python-env .

fetch-data:
	docker run --rm -v ${PWD}:/home/jovyan/work python-env ./scripts/fetch-data.sh

interact:
	docker run --rm -it -v ${PWD}:/home/jovyan/work python-env /bin/bash

cp-modules:
	docker cp ../utils python-env:/home/jovyan/work/notebooks/
	docker cp ../kaggle_secrets python-env:/home/jovyan/work/notebooks/
	docker cp ../kaggle_secrets python-env:/home/jovyan/work/notebooks/

dev:
	make build
	make fetch-data
	make notebook
