#!/bin/bash


# default values
SEED=0
EPOCHS=1
NUM=8
BUDGET=3
EPLEN=20
NUMEPISODE=30
DATA="synthetic"

while getopts s:e:n:b:h:t:d: option
do
case "${option}"
in
s) SEED=${OPTARG};;
e) EPOCHS=${OPTARG};;
# vary settings
n) NUM=${OPTARG};;
b) BUDGET=${OPTARG};;
h) EPLEN=${OPTARG};;
t) NUMEPISODE=${OPTARG};;
d) DATA=${OPTARG};;
esac
done

echo "data is" $DATA
