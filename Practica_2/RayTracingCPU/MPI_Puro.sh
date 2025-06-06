#!/bin/bash

EXECUTABLE="./raytracer.exe"
SIZES=("32 32" "64 64" "128 128" "256 256" "512 512")
PROCESOS=(1 2 3 4 8 16)

for np in "${PROCESOS[@]}"; do
    echo -e "\n"
    echo "Ejecutando con $np procesos"

    for size in "${SIZES[@]}"; do        
        if [ "$np" -eq 8 ]; then
            mpirun --oversubscribe -np $np $EXECUTABLE $size
        else
            mpirun -np $np $EXECUTABLE $size
        fi
    done
done
