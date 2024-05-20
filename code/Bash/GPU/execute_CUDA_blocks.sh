#! /bin/bash

# Check the number of parameters
if [ "$#" -ne 5 ]; then
    echo "Error: Illegal number of parameters, please provide:"
    echo "1) Executable path"
    echo "2) Number of iterations"
    echo "3) Max number of CUDA Blocks"
    echo "4) Step used for incrementing the number of CUDA Blocks"
    echo "5) Number to be Factorized"
    exit 1
fi

# PARAMETERS:

# Path of the executable
EXECUTABLE=$1
# Number of iterations
NUM_ITERATIONS=$2
# Maximum number of CUDA Blocks
MAX_CUDA_BLOCKS=$3
# Step used for incrementing the number of CUDA Blocks
STEP_CUDA_BLOCKS=$4
# Number to be factorized
NUMBER=$5

# Local variables:
# Current Time
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
# Output file
OUTPUT_FILE="output_${TIME}.csv"

echo "Executing the program with the following parameters:"
echo ""
echo "Executable Path: ${EXECUTABLE}"
echo "Number of iterations: ${NUM_ITERATIONS}"
echo "Max number of CUDA Blocks: ${MAX_CUDA_BLOCKS}"
echo "Step of the number of CUDA Blocks: ${STEP_CUDA_BLOCKS}"
echo "Number to be factorized: ${NUMBER}"
echo ""

# Creating the structure of the output file (CSV)
echo "num_CUDA_blocks,iteration,GPU_execution_time,TOTAL_execution_time" > ${OUTPUT_FILE}

# Execute the program
for ((NUM_CUDA_BLOCKS = ${STEP_CUDA_BLOCKS}; NUM_CUDA_BLOCKS <= ${MAX_CUDA_BLOCKS}; NUM_CUDA_BLOCKS += ${STEP_CUDA_BLOCKS})); do
    for ((i = 0; i < ${NUM_ITERATIONS}; i++)); do
        echo "Running iteration ${i} with ${NUM_CUDA_BLOCKS} CUDA Blocks"
        
        echo -n "${NUM_CUDA_BLOCKS},${i}," >> ${OUTPUT_FILE}
        ./${EXECUTABLE} ${NUM_CUDA_BLOCKS} ${NUMBER} 0 >> ${OUTPUT_FILE} 
    done
done