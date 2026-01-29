SHELL := /bin/bash

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

.PHONY: prepare build up down logs test

prepare:
	mkdir -p data
	@if [ ! -w data ]; then sudo chown $(USER_ID):$(GROUP_ID) data; fi
	@chmod 775 data

build:
	$(MAKE) prepare
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose build

up:
	$(MAKE) prepare
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose up --build -d

down:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose down

logs:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose logs -f

test:
	$(MAKE) prepare
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose --profile test run --rm --build qwen-tts-test
