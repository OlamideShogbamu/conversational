#!/usr/bin/env bash

# This tags and uploads an image to Docker Hub

#This is your Docker ID/path
dockerpath=herbehordeun/registry.heroku.com/dry-peak-46846/web
# Run the Docker Hub container with kubernetes

kubectl run registry.heroku.com/dry-peak-46846/web\
    --image=$dockerpath\
    --port=5000 --labels app=registry.heroku.com/dry-peak-46846/web


# List kubernetes pods
kubectl get pods

# Forward the container port to a host
kubectl port-forward registry.heroku.com/dry-peak-46846/web 5000:5000