#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;

/**
 * @brief Struct to store prime factors and their exponents
 */
struct factor_exponent {
    unsigned long long factor;
    int exponent;
};

/**
 * @brief Kernel function to check if a number is prime
 *
 * @param n number to check if prime
 * @return true if prime, false otherwise
 */
__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (unsigned long long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

/**
 * @brief Kernel function to find prime factors in a range
 *
 * @param start integer to start from
 * @param end integer to end at
 * @param num number to find prime factors of
 * @param primes vector to store prime factors
 */
__global__ void findPrimesInRange(unsigned long long start, unsigned long long end, unsigned long long num, factor_exponent* primes, int* primes_count) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long step = gridDim.x * blockDim.x;

    for (unsigned long long i = start + idx; i <= end; i += step) {
        if ((num % i) == 0) {
            int exponent = 0;
            while (num % i == 0) {
                exponent++;
                num /= i;
            }
            if (isPrime(i)) {
                int index = atomicAdd(primes_count, 1);
                primes[index] = { i, exponent };
            }
        }
    }
}

/**
 * @brief Main function for parallel factorization, using trial division algorithm in CUDA
 *
 * @param num number to find prime factors of
 * @param numThreads number of threads to use
 * @return vector<factor_exponent> vector of prime factors
 */
vector<factor_exponent> parallelTrialDivision(unsigned long long num, int numThreads) {
    vector<factor_exponent> primes;
    factor_exponent* d_primes;
    int* d_primes_count;

    cudaMalloc((void**)&d_primes, sizeof(factor_exponent) * num);
    cudaMalloc((void**)&d_primes_count, sizeof(int));
    cudaMemset(d_primes_count, 0, sizeof(int));

    // checking if the number is divisible by 2
    if (num % 2 == 0) {
        int exponent = 0;
        while (num % 2 == 0) {
            exponent++;
            num /= 2;
        }
        primes.push_back({ 2, exponent });
    }

    unsigned long long sqrt_num = (unsigned long long)sqrt(num);
    unsigned long long range = sqrt_num / numThreads;

    // define the start and end of the range for the first thread
    unsigned long long start = 3;
    unsigned long long end = (range % 2 == 0) ? range + 1 : range;

    // Launch kernel to find primes
    findPrimesInRange<<<numThreads, 256>>>(start, end, num, d_primes, d_primes_count);

    cudaDeviceSynchronize();

    int primes_count;
    cudaMemcpy(&primes_count, d_primes_count, sizeof(int), cudaMemcpyDeviceToHost);
    primes.resize(primes_count);
    cudaMemcpy(primes.data(), d_primes, sizeof(factor_exponent) * primes_count, cudaMemcpyDeviceToHost);

    // if primes is empty than the number is prime so add it to the vector
    if (primes.empty()) {
        primes.push_back({ num, 1 });
    } else {
        unsigned long long product = 1;
        for (factor_exponent& prime : primes) {
            for (int i = 0; i < prime.exponent; i++) {
                product *= prime.factor;
            }
        }
        if (product != num) {
            primes.push_back({ num / product, 1 });
        }
    }

    cudaFree(d_primes);
    cudaFree(d_primes_count);

    return primes;
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        cout << "Please provide:\n1) Number of Threads\n2) Number to be Factorized\n3) 0 or 1 to execute the program in BASH or USER mode." << endl;
        return 1;
    }

    int NUM_THREADS = atoi(argv[1]);
    unsigned long long NUMBER = stoull(argv[2]);
    bool EXECUTION_MODE = atoi(argv[3]);

    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    vector<factor_exponent> factors = parallelTrialDivision(NUMBER, NUM_THREADS);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    if (EXECUTION_MODE) {
        cout << "Time taken: " << duration.count() << " milliseconds." << endl;

        cout << NUMBER << " = ";
        for (vector<factor_exponent>::iterator it = factors.begin(); it != factors.end(); ++it) {
            cout << it->factor << "^" << it->exponent;
            if (next(it) != factors.end()) {
                cout << " * ";
            }
        }
        cout << endl;
    } else {
        cout << duration.count() << endl;
    }

    return 0;
}