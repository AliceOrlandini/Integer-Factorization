#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

struct factor_exponent {
    unsigned long long factor;
    int exponent;
};

__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 3 == 0) return false;
    for (unsigned long long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

__global__ void findPrimesInRange(unsigned long long start, unsigned long long end, unsigned long long num, factor_exponent* primes, int* primes_count) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;

    for (unsigned long long i = start + 2 * tid; i <= end; i += 2 * stride) {
        if (num % i == 0) {
            int exponent = 0;
            unsigned long long temp_num = num;
            while (temp_num % i == 0) {
                exponent++;
                temp_num /= i;
            }

            if (isPrime(i)) {
                int index = atomicAdd(primes_count, 1);
                primes[index].factor = i;
                primes[index].exponent = exponent;
            }
        }
    }
}

vector<factor_exponent> parallelTrialDivision(unsigned long long num, int numThreads) {
    vector<factor_exponent> primes;
    factor_exponent* d_primes;
    int* d_primes_count;
    unsigned long long old_num = num;

    int primes_count = 0;

    cudaMalloc(&d_primes, sizeof(factor_exponent) * numThreads * 100);
    cudaMalloc(&d_primes_count, sizeof(int));
    cudaMemcpy(d_primes_count, &primes_count, sizeof(int), cudaMemcpyHostToDevice);

    if (num % 2 == 0) {
        int exponent = 0;
        while (num % 2 == 0) {
            exponent++;
            num /= 2;
        }
        primes.push_back({2, exponent});
    }

    unsigned long long sqrt_num = (unsigned long long)sqrt(num);
    unsigned long long range = sqrt_num / numThreads;

    unsigned long long start = 3;
    unsigned long long end = sqrt_num;

    int blockSize = 256;
    int numBlocks = (range + blockSize - 1) / blockSize;
    findPrimesInRange<<<numBlocks, blockSize>>>(start, end, num, d_primes, d_primes_count);

    cudaDeviceSynchronize();

    cudaMemcpy(&primes_count, d_primes_count, sizeof(int), cudaMemcpyDeviceToHost);

    factor_exponent* h_primes = new factor_exponent[primes_count];
    cudaMemcpy(h_primes, d_primes, sizeof(factor_exponent) * primes_count, cudaMemcpyDeviceToHost);

    for (int i = 0; i < primes_count; ++i) {
        primes.push_back(h_primes[i]);
    }

    delete[] h_primes;
    cudaFree(d_primes);
    cudaFree(d_primes_count);

    if (primes.empty()) {
        primes.push_back({num, 1});
    } else {
        unsigned long long product = 1;
        for (auto& fe : primes) {
            for (int i = 0; i < fe.exponent; i++) {
                product *= fe.factor;
            }
        }
        if (product != old_num) {
            primes.push_back({old_num / product, 1});
        }
    }

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
        for (auto it = factors.begin(); it != factors.end(); ++it) {
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