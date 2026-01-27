SHELL := /bin/bash

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

.PHONY: build up down logs

build:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) sudo -E docker compose build

up:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) sudo -E docker compose up --build -d

down:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) sudo -E docker compose down

logs:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) sudo -E docker compose logs -f
