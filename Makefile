# -*- mode: makefile-gmake; mode: pabbrev; mode: electric-pair -*-
include config.mk

VALIDATE_ENDPOINT := https://api.up42.com/validate-schema/block
REGISTRY := registry.up42.com
CURL := curl
DOCKER := docker

build-image-esa-snap:
	$(DOCKER) build -f $(ESA_SNAP_DOCKERFILE) -t up42-esa-snap .

build-image-up42-snap: 
	$(DOCKER) build -f $(UP42_SNAP_DOCKERFILE) -t up42-snap .

build: $(MANIFEST_JSON) build-image-esa-snap build-image-up42-snap
	$(DOCKER) build --build-arg manifest="$$(cat $<)" -f $(UP42_DOCKERFILE) -t $(REGISTRY)/$(UID)/$(DOCKER_TAG) .

validate: $(MANIFEST_JSON)
	$(CURL) -X POST -H 'Content-Type: application/json' -d @$^ $(VALIDATE_ENDPOINT) 

push:	
	$(DOCKER) push $(REGISTRY)/$(UID)/$(DOCKER_TAG)

login:
	$(DOCKER) login -u $(USER) https://$(REGISTRY)

run: $(JOB_CONFIG) build-all
	$(DOCKER) run -e UP42_TASK_PARAMETERS="$$(cat $<)" $(DOCKER_RUN_OPTIONS) $(DOCKER_TAG) 

.PHONY: build-image-esa-snap build-image-up42-snap build login push run
