# Integer-Factorization
A Computer Architecture Project for the 2023/2024 academic year.

![Integer-Factorization](Slides/Resources/IntegerFactorization.png)

## Overview
This repository contains implementations of a parallel trial division algorithm for factorizing numbers. The repository includes both the initial and optimized versions, as well as a version utilizing CUDA for GPU acceleration.

## Project Structure

The project is organized into the following directories:

- **`Code/`**: Contains all the source code for the project.
    
    - **`CPU/`**: Contains CPU-based implementations of the trial division algorithm and related utilities.

        - **`Bash/`**: Bash scripts for running CPU-based implementations and generating data for analysis.
            - `execute_MaxNumber.sh`: Executes an executable NUM_ITERATIONS times, varying the number to be factorized from STEP to MAX_NUMBER in increments of STEP. Outputs results to a CSV file.
            - `execute_MaxThreads.sh`: Executes an executable NUM_ITERATIONS times, varying the number of threads from 1 to MAX_THREADS. Outputs results to a CSV file.
            - `execute_OneConfiguration.sh`: Executes an executable with a specific configuration NUM_ITERATIONS times. Outputs results to a CSV file.

        - **`Batch/`**: Batch scripts for running CPU-based implementations and generating data for analysis.
            - `execute_MaxNumber.bat`: Executes an executable NUM_ITERATIONS times with NUM_THREADS threads, varying the number to be factorized from STEP to MAX_NUMBER in increments of STEP. Outputs results to a CSV file.
            - `execute_MaxThreads.bat`: Executes an executable NUM_ITERATIONS times, varying the number of threads from 1 to MAX_THREADS. Outputs results to a CSV file.

        - **`C++/`**: C++ source code for the CPU-based implementations.
            - `First_TrialDivision.cpp`: Initial version of the parallel trial division algorithm.
            - `Optimized_TrialDivision.cpp`: Optimized version of the parallel trial division algorithm.
            - `Optimized_ThreadPool_Affinity_TrialDivision.cpp`: Further optimized version using a thread pool and thread affinity to manage and assign threads to different cores.
            - **`Debug/`**: Versions of the CPU-based implementations used for debugging.
                - `Debug_Optimized_ThreadPool_Affinity_TrialDivision.cpp`: Simplified version of the thread pool implementation for testing and comparison.

        - **`Insights/`**: Insights obtained from analyzing the CPU-based implementations.
            - `cpu.md`: Documentation of insights from the CPU-based implementations analysis.

        - **`Python/`**: Python scripts for performance and scalability analysis.
            - **`Performance/`**: Performance analysis experiments.
            - **`Scalability/`**: Scalability analysis experiments.

    - **`GPU/`**: Contains CUDA-based implementations of the trial division algorithm and related utilities.

        - **`Bash/`**: Bash scripts for running CUDA-based implementations and generating data for analysis.
            - `execute_CUDA_blocks.sh`: Executes a CUDA program NUM_ITERATIONS times, varying the number of CUDA blocks from STEP_CUDA_BLOCKS to MAX_CUDA_BLOCKS. Outputs results to a CSV file.

        - **`CUDA C/`**: CUDA C source code for the GPU-based implementations.
            - `First_TrialDivision.cu`: Initial version of the parallel trial division algorithm using CUDA to exploit GPU parallelism.
            - `Optimized_TrialDivision.cu`: Optimized version of the parallel trial division algorithm using CUDA for improved GPU parallelism.

        - **`Insights/`**: Insights obtained from analyzing the CUDA-based implementations.
            - `gpu.md`: Documentation of insights from the CUDA-based implementations analysis.

        - **`Python/`**: Python scripts for performance analysis of CUDA implementations.
            - **`Performance/`**: Performance analysis experiments.

- **`Resources/`**: Contains papers, articles, and other resources used during the project.

- **`Slides/`**: Contains slides for the project presentation.
    - `Palette.txt`: Color palette used in the presentation.
    - `SlidesCA_IntegerFactorization.pdf`, `SlidesCA_IntegerFactorization.pptx`: Current presentation slides in PDF and PPTX formats.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
