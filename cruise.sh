#!/bin/bash
#Time period to run the simulation for
START=100
END=1000

## save $START, just in case if we need it later ##
i=$START

while [ $i -lt $END ]
do
    echo "************"
    echo "Running MUMA"
    echo "************"
    python muma.py $i
    i=`expr $i + 100`
done

while [ $i -lt $END ]
do
    echo "******************"
    echo "Running Equal MUMA"
    echo "******************"
    python equal.py $i
    i=`expr $i + 100`
done

while [ $i -lt $END ]
do
    echo "*************"
    echo "Running AORTA"
    echo "*************"
    python aorta.py $i
    i=`expr $i + 100`
done