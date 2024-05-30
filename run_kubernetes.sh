#!/usr/bin/env bash

# This tags and uploads an image to Docker Hub

#This is your Docker ID/path
dockerpath=herbehordeun/conversational-bot-app
# Run the Docker Hub container with kubernetes

kubectl run conversational-bot-app\
    --image=$dockerpath\
    --port=5000 --labels app=conversational-bot-app


# List kubernetes pods
kubectl get pods

# Forward the container port to a host
kubectl port-forward conversational-bot-app 5000:5000