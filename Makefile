DOCKER_NAME=udi_docker
srcdir ?= $(shell pwd)

build:
	docker build -f Dockerfile -t $(DOCKER_NAME) .

run:
	docker run  -it  -p 4599:4567  -v $(srcdir):/capstone $(DOCKER_NAME) 

attach:
	docker exec  -it  $(DOCKER_NAME)
