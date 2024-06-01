#!/usr/bin/env bash
# This file tags and uploads an image to Docker Hub

# Assumes that an image is built via `run_docker.sh`

# Step 1:
# Create dockerpath
dockerpath=herbehordeun/registry.heroku.com/dry-peak-46846/web

# Step 2:
# Authenticate & tag
docker login --username=herbehordeun
docker image tag registry.heroku.com/dry-peak-46846/web $dockerpath
echo "Docker ID and Image: $dockerpath"

# Step 3:
docker push $dockerpath:latest