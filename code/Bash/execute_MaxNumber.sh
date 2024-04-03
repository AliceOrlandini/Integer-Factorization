#! /bin/bash

# Check the number of parameters
if [ "$#" -ne 5 ]; then
    echo "Error: Illegal number of parameters"
    exit 1
fi

# Parameters:
# Path of the executable
EXECUTABLE=$1
# Number of iterations
NUM_ITERATIONS=$2
# Number of threads used in the execution
NUM_THREADS=$3
# MAX Number to be factorized
MAX_NUMBER=$4
# Step used for incrementing the number to be factorized
STEP=$5

# Local variables:
# Current Time
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
# Output file
OUTPUT_FILE="output_${TIME}.csv"

echo "Executing the program with the following parameters:"
echo ""
echo "Executable Path: ${EXECUTABLE}"
echo "Number of iterations: ${NUM_ITERATIONS}"
echo "Number of threads: ${NUM_THREADS}"
echo "Max number to be factorized: ${MAX_NUMBER}"
echo "Step of the number: ${STEP}"
echo ""

# Creating the structure of the output file (CSV)
echo "number,iteration,execution_time" > ${OUTPUT_FILE}

# Execute the program
for ((NUMBER = ${STEP}; NUMBER <= ${MAX_NUMBER}; NUMBER += ${STEP})); do
    for ((i = 0; i < ${NUM_ITERATIONS}; i++)); do
        echo "Running iteration ${i} with the number ${NUMBER}"
        
        echo -n "${NUMBER},${i}," >> ${OUTPUT_FILE}
        ./${EXECUTABLE} ${NUM_THREADS} ${NUMBER} 0 >> ${OUTPUT_FILE} 
    done
done