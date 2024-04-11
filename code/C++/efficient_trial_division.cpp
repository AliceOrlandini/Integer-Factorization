#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <math.h>

using namespace std;
mutex mtx; // mutex for output synchronization
struct factor_exponent {
    unsigned long long factor;
    int exponent;
};

unsigned long long modulo(unsigned long long base, unsigned long long exponent, unsigned long long mod) {
    int result = 1;
    base = base % mod;
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result = (result * base) % mod;   
        }
        exponent = exponent >> 1;
        base = (base * base) % mod;
    }
    return result;
}

bool fermatTest(unsigned long long n, int iterations) {
    if (n <= 1 || n == 4){
        return false;
    }
    if (n <= 3) {
        return true;
    }

    for (int i = 0; i < iterations; i++) {
        unsigned long long a = 2 + rand() % (n - 4);
        if (modulo(a, n - 1, n) != 1) { 
            return false;
        }
    }
    return true;
}

/**
 * @brief Function to check if a number is prime
 *
 * @param n number to check if prime
 * @return true if prime, false otherwise
 */
bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    // if (n % 2 == 0 || n % 3 == 0) return false;
    if (n % 3 == 0) return false;
    for (unsigned long long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

/**
 * @brief Trial division function to find prime factors in a range
 *
 * @param start integer to start from
 * @param end integer to end at
 * @param num number to find prime factors of
 * @param primes vector to store prime factors
 */
void findPrimesInRange(unsigned long long start, unsigned long long end, unsigned long long num, vector<factor_exponent>& primes) {

    // check all numbers in the range
    for (unsigned long long i = start; i <= end; i += 2) {
        // if i in the range is prime and num is divisible by i
        // add it to the primes vector
        if (fermatTest(i, 10)) {

            // perform the isPrime check only if the Fermat test has passed
            if(isPrime(i) && (num % i) == 0) {

                // continue dividing as long as possible
                // this way we avoid adding the same factor multiple times
                int exponent = 0;
                while (num % i == 0) { 
                    exponent++;
                    num /= i;
                }

                // lock the mutex
                {
                    lock_guard<mutex> lock(mtx);
                    primes.push_back({ i, exponent });
                }
            }
        }
    }
}

/**
 * @brief Main function for parallel factorization, using trial division
 *
 * @param num number to find prime factors of
 * @param numThreads number of threads to use
 * @return vector<factor_exponent> vector of prime factors
 */
vector<factor_exponent> parallelTrialDivision(unsigned long long num, int numThreads) {

    vector<factor_exponent> primes;
    vector<thread> threads;

    // number of threads to be used for parallelization
    // const int numThreads = thread::hardware_concurrency();

    unsigned long long old_num = num;

    // Checking in advance if the number is divisible by 2
    if (num % 2 == 0) {
        int exponent = 0;
        while (num % 2 == 0) {
            exponent++;
            num /= 2;
        }
        primes.push_back({ 2, exponent });
    }
    // Now the interval to check is nearly halved
    // as checking divisibility by even numbers is not needed

    unsigned long long sqrt_num = (unsigned long long) sqrt(num);

    // divide the work equally among the threads
    unsigned long long range = sqrt_num / numThreads;

    // define the start and end of the range for the first thread
    unsigned long long start = 3;
    unsigned long long end = (range % 2 == 0) ? range + 1 : range;

    // create and start the threads
    for (int i = 0; i < (numThreads - 1); ++i) {
        threads.emplace_back(findPrimesInRange, start, end, num, ref(primes));

        // Print thread identifier
        // cout <<"Thread["<<i<<"] ID: " << threads[i].get_id() << endl;

        // update the start and end for the next thread
        start = end + 2;

        if(range % 2 == 0){
            end = start + range;
        }
        else{
            end = start + range - 1;
        }
    }
    end = sqrt_num;

    // Last thread is the main thread
    findPrimesInRange(start, end, num, primes);

    // wait for the other threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    // If primes is empty (i.e. the number is prime)
    if(primes.empty()) {
        primes.push_back({ num, 1 });
    } else {
        // Check if all the factors have been found 
        // (otherwise a prime factor larger than the 
        // square root of the number is missing)

        unsigned long long product = 1;
        for (auto it = primes.begin(); it != primes.end(); ++it) {
            product *= pow(it->factor, it->exponent);
        }
        if (product != old_num) {
            primes.push_back({ old_num / product, 1 });
        }
    }    

    return primes;
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        cout << "Please provide: Number of Threads, Number to be Factorized and 0 or 1 to execute the program in BASH or USER mode." << endl;
        return 1;
    }

    // get the number of threads from the command line argument
    int NUM_THREADS = atoi(argv[1]);

    // get the number from the command line argument
    unsigned long long NUMBER = stoull(argv[2]);

    // get the mode (0: bash, 1: user)
    bool EXECUTION_MODE = atoi(argv[3]);

    // start measuring time
    // auto start = chrono::high_resolution_clock::now();
    auto start = chrono::steady_clock::now();
    // chrono::steady_clock::time_point start = chrono::high_resolution_clock::now();

    // find the prime factors of the number
    vector<factor_exponent> factors = parallelTrialDivision(NUMBER, NUM_THREADS);

    // stop measuring time
    // auto end = chrono::high_resolution_clock::now();
    auto end = chrono::steady_clock::now();
    // chrono::steady_clock::time_point end = chrono::high_resolution_clock::now();


    // calculate the time duration
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    // chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    if (EXECUTION_MODE) {
        cout << "Time taken: " << duration.count() << " milliseconds." << endl;

        cout << NUMBER << " = ";
        for (auto it = factors.begin(); it != factors.end(); ++it) {
            cout << it->factor << "^" << it->exponent;
            // print a * between factors except for the last one
            if (next(it) != factors.end()) {
                cout << " * ";
            }
        }
        cout << endl;
    }
    else {
        cout << duration.count() << endl;
    }

    return 0;
}
