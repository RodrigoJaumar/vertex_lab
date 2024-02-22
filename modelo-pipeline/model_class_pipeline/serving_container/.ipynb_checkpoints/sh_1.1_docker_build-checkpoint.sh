#!/bin/bash     
PROJECT_ID="careful-acumen-414922"
REGION="europe-west2"
REPOSITORY="houseprice"
IMAGE='serving_image'
IMAGE_TAG='serving_image:latest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
