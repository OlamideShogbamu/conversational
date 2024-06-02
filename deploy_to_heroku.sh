#!/usr/bin/env bash

# Build the Docker image
docker build -t registry.heroku.com/registry.heroku.com/dry-peak-46846/web .

# Push the Docker image to Heroku
docker push registry.heroku.com/registry.heroku.com/dry-peak-46846/web