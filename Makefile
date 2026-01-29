SHELL := /bin/bash

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

.PHONY: build up down logs test

build:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose build

up:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose up --build -d

down:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose down

logs:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose logs -f

test:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose --profile test run --rm --build qwen-tts-test