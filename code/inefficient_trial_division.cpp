#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

using namespace std;
mutex mtx; // mutex for output synchronization

/**
 * @brief Function to check if a number is prime (inefficiently)
 * 
 * @param n number to check if prime
 * @return true if prime, false otherwise
 */
bool isPrime(int n) {
    
    bool is_prime = true;
    
    // 0 and 1 are not prime numbers
    if (n == 0 || n == 1) {
        is_prime = false;
    }

    // check if n is divisible by any number from 2 to n/2
    // because if n is divisible by any number greater than n/2
    // then it would also be divisible by a number less than n/2
    // which we would have already checked
    for (int i = 2; i <= n/2; ++i) {
        if (n % i == 0) {
            is_prime = false;
            break;
        }
    }

    return is_prime;
}

/**
 * @brief Trial division function to find prime factors in a range
 * 
 * @param start integer to start from
 * @param end integer to end at
 * @param num number to find prime factors of
 * @param primes vector to store prime factors
 */
void findPrimesInRange(int start, int end, int num, vector<int>& primes) {
    
    // check all numbers in the range
    for (int i = start; i <= end; ++i) {
        // if i in the range is prime and num is divisible by i
        // add it to the primes vector
        if (isPrime(i) && num % i == 0) {
            // I create a block to lock the mutex
            // so that the lock_guard is destroyed after the block
            {
                // lock the mutex
                lock_guard<mutex> lock(mtx);

                // add the prime factor to the vector
                primes.push_back(i);
            } 
            // continue dividing as long as possible
            // this way we avoid adding the same factor multiple times
            while (num % i == 0) {
                num /= i;
            }
        }
    }
}

/**
 * @brief Main function for parallel factorization, using trial division
 * 
 * @param num number to find prime factors of
 * @return vector<int> vector of prime factors
 */
vector<int> parallelTrialDivision(int num) {
    
    vector<int> primes;
    vector<thread> threads;

    // number of threads to be used for parallelization
    const int numThreads = thread::hardware_concurrency();

    // divide the work equally among the threads
    int range = num / numThreads;

    // define the start and end of the range for the first thread
    int start = 2;
    int end = range;

    // create and start the threads
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(findPrimesInRange, start, end, num, ref(primes));

        // update the start and end for the next thread
        start = end + 1;

        // if it is the second last thread, the end is the number
        // otherwise, the end is the start plus the range minus 1
        if(i == numThreads - 2) { // why -2? why not -1?
            end = num;
        } else {
            end = start + range - 1;
        }

    }

    // wait for the threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    return primes;
}

int main(int argc, char *argv[]) {
    
    if (argc != 2) {
        cout << "Please provide a number as argument." << endl;
        return 1;
    }

    // get the number from the command line argument
    int num = atoi(argv[1]);

    // find the prime factors of the number
    vector<int> factors = parallelTrialDivision(num);

    cout << "Prime numbers of " << num << " are: ";
    for (int factor : factors) {
        cout << factor << " ";
    }
    cout << endl;

    return 0;
}
