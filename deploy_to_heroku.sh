#!/usr/bin/env bash

# Build the Docker image
docker build -t registry.heroku.com/conversational-bot-app/web .

# Push the Docker image to Heroku
docker push registry.heroku.com/conversational-bot-app/web