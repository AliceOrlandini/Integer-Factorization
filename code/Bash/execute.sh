#! /bin/bash

# Parameters
# Path of the executable
EXECUTABLE=$1
# Number of iterations
NUM_ITERATIONS=$2
# Number of threads
NUM_THREADS=$3
# Number to be factorized
NUMBER=$4
# Current Time
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
# Output file
OUTPUT_FILE="output_${TIME}.txt"

echo "Executing the program with the following parameters:"
echo ""
echo "Executable Path: ${EXECUTABLE}"
echo "Number of iterations: ${NUM_ITERATIONS}"
echo "Number of threads: ${NUM_THREADS}"
echo "Number to be factorized: ${NUMBER}"
echo ""

# Creating the structure of the output file (CSV)
echo "iteration, execution_time" > ${OUTPUT_FILE}

# Execute the program
for ((i = 0; i < ${NUM_ITERATIONS}; i++)); do
    echo "Running iteration ${i}"
    
    echo -n "${i}, " >> ${OUTPUT_FILE}
    ./${EXECUTABLE} ${NUM_THREADS} ${NUMBER} 0 >> ${OUTPUT_FILE} 
done