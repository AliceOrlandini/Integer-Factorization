#! /bin/bash

# This script is used to execute an executable for NUM_ITERATIONS times, changing the number of threads
# from 1 to MAX_THREADS. The output of the program is saved in a CSV.

# Check the number of parameters
if [ "$#" -ne 4 ]; then
    echo "Error: Illegal number of parameters, please provide:"
    echo " 1. Executable path"
    echo " 2. Number of iterations"
    echo " 3. Max number of threads"
    echo " 4. Number to be factorized"
    exit 1
fi

# PARAMETERS

# Path of the executable
EXECUTABLE=$1
# Number of iterations
NUM_ITERATIONS=$2
# Maximum number of threads to be tested
MAX_THREADS=$3
# Number to be factorized
NUMBER=$4

# Local variables:
# Current Time
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
# Output file
OUTPUT_FILE="output_${TIME}.csv"

echo "Executing the program with the following parameters:"
echo ""
echo "Executable Path: ${EXECUTABLE}"
echo "Number of iterations: ${NUM_ITERATIONS}"
echo "Number of threads: ${MAX_THREADS}"
echo "Number to be factorized: ${NUMBER}"
echo ""

# Creating the structure of the output file (CSV)
echo "num_threads,iteration,execution_time" > ${OUTPUT_FILE}

# Execute the program
for ((NUM_THREADS = 1; NUM_THREADS <= ${MAX_THREADS}; NUM_THREADS++)); do
    for ((i = 0; i < ${NUM_ITERATIONS}; i++)); do
        echo "Running iteration ${i} with ${NUM_THREADS} threads"
        
        echo -n "${NUM_THREADS},${i}," >> ${OUTPUT_FILE}
        ./${EXECUTABLE} ${NUM_THREADS} ${NUMBER} 0 >> ${OUTPUT_FILE} 
    done
done