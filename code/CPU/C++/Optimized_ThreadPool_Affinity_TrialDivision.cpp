/**
 * [OPTIMIZED + THREAD POOL + AFFINITY VERSION]
 * In this optimized version of the parallel trial division algorithm,
 * a thread pool is used to manage the threads. The thread pool is created
 * before the time measurement starts to avoid counting the time taken to
 * create the threads in the total time taken to execute the program.
 * Furthermore, each thread is assigned to a different core using thread affinity.
 * 
 */

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <math.h>
#include <queue>
#include <condition_variable>
#include <functional>
#include <future>
#include <windows.h>

using namespace std;

/*
-----------------------------------------------------------------------------------------
--------------------------------- (START) THREAD POOL -----------------------------------
-----------------------------------------------------------------------------------------
 */
class ThreadPool {

public:
    ThreadPool(size_t);

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> future<typename result_of<F(Args...)>::type>;

    void wait();

    ~ThreadPool();

private:
    vector<thread> threads;

    queue<function<void()>> tasks;

    mutex queueMutex;

    condition_variable condition;

    bool stop;

};

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {

    for (size_t i = 1; i < numThreads + 1; ++i) {
        threads.emplace_back(
            [this] {

                function<void()> task;
                {
                    unique_lock<mutex> lock(this->queueMutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }
                    task = move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
        
            }
        );

        DWORD_PTR dw = SetThreadAffinityMask(threads.back().native_handle(), DWORD_PTR(1) << i);
        if (dw == 0) {
            DWORD dwErr = GetLastError();
            cerr << "SetThreadAffinityMask failed, GLE=" << dwErr << '\n';
        }

    }
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> future<typename result_of<F(Args...)>::type> {
    using returnType = decltype(f(args...));
    auto task = make_shared<packaged_task<returnType()>>(bind(forward<F>(f), forward<Args>(args)...));
    future<returnType> res = task->get_future();
    {
        unique_lock<mutex> lock(queueMutex);
        if (stop) {
            throw runtime_error("enqueue on stopped ThreadPool");
        }
        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

void ThreadPool::wait() {
    for (thread& worker : threads) {
        worker.join();
    }
}

ThreadPool::~ThreadPool() {
    {
        unique_lock<mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
}
/*
-----------------------------------------------------------------------------------------
----------------------------------- (END) THREAD POOL -----------------------------------
-----------------------------------------------------------------------------------------
 */



// Mutex for synchronizing access to shared data
mutex mtx;

/**
 * @brief Struct to store prime factors and their exponents
 */
struct factor_exponent {
    unsigned long long factor; // Prime factor
    int exponent; // Exponent of the prime factor
};

/**
 * @brief Checks if a number is prime
 * 
 * @param n Number to check for primality
 * @return true if the number is prime, false otherwise
 */
bool isPrime (unsigned long long n) {
    // Numbers less than or equal to 1 are not prime
    if (n <= 1) return false;
    // Numbers 2 and 3 are prime
    if (n <= 3) return true;
    // Multiples of 3 are not prime
    if (n % 3 == 0) return false;
    // Check for factors from 5 to sqrt(n)
    for (unsigned long long i = 5; i * i <= n; i += 6) {
        // If divisible by i or (i + 2), n is not prime
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

/**
 * @brief Finds prime factors of a number in a given range
 * 
 * @param start Starting number of the range
 * @param end Ending number of the range
 * @param num Number to be factorized
 * @param primes Vector to store prime factors and their exponents
 */
void findPrimesInRange (unsigned long long start, unsigned long long end, unsigned long long num, vector<factor_exponent>& primes) {

    // (START) DEBUG
    // {
    //     lock_guard<mutex> lock(mtx);
    //     // Get the thread id
    //     thread::id this_id = this_thread::get_id();
    //     cout << "#START: Thread ID: " << this_id << " is running on core: " << GetCurrentProcessorNumber() << endl;
    // }
    // (END) DEBUG


    // Iterate through the range
    for (unsigned long long i = start; i <= end; i += 2) {
        // Check if num is divisible by i
        if ((num % i) == 0) {
            int exponent = 0;
            // Count the exponent of the prime factor
            while (num % i == 0) { 
                exponent++;
                // Divide num by i as long as possible
                // this avoids adding the same prime factor multiple times
                num /= i;
            }
            // Lock the mutex to save the prime factor and its exponent
            if (isPrime(i)) {
                lock_guard<mutex> lock(mtx);
                primes.push_back({ i, exponent });
            }
        }
    }

    // (START) DEBUG
    // {
    //     lock_guard<mutex> lock(mtx);
    //     // Get the thread id
    //     thread::id this_id = this_thread::get_id();
    //     cout << "#END: Thread ID: " << this_id << " is running on core: " << GetCurrentProcessorNumber() << endl;
    // }
    // (END) DEBUG

}


/**
 * @brief Parallel factorization using trial division
 * 
 * @param num Number to be factorized
 * @param numThreads Number of threads to use
 * @param pool ThreadPool object to manage the threads
 * @return vector<factor_exponent> Vector of prime factors and their exponents
 */
vector<factor_exponent> parallelTrialDivision (unsigned long long num, int numThreads, ThreadPool& pool) {

    // (START) DEBUG
    // thread::id this_id = this_thread::get_id();
    // cout << "#MAIN: Thread ID: " << this_id << " is running on core: " << GetCurrentProcessorNumber() << endl << endl;
    // (END) DEBUG

    // Vector to store prime factors and their exponents
    vector<factor_exponent> primes;
    // Vector to store threads
    vector<thread> threads;

    // Store the original number for later use
    unsigned long long old_num = num;

    // Checking in advance if the number is divisible by 2
    // To avoid checking even numbers in the loop
    if (num % 2 == 0) {
        int exponent = 0;
        while (num % 2 == 0) {
            exponent++;
            num /= 2;
        }
        // Here we don't need to lock the mutex as we are in the main thread
        primes.push_back({ 2, exponent });
    }

    // Now the interval to check is nearly halved
    // As checking divisibility by even numbers is not needed

    // Calculate the square root of num
    unsigned long long sqrt_num = (unsigned long long) sqrt(num);

    // Calculate the range each thread will process
    unsigned long long range = sqrt_num / numThreads;

    // Starting number for the first thread
    unsigned long long start = 3;
    // Ending number for the first thread
    unsigned long long end = (range % 2 == 0) ? range + 1 : range;


    for (int i = 0; i < (numThreads - 1); ++i) {

        // enqueue the task to the thread pool
        pool.enqueue(findPrimesInRange, start, end, num, ref(primes));

        // Update the start and end for the next thread
        start = end + 2;
        if (range % 2 == 0) {
            end = start + range;
        }
        else {
            end = start + range - 1;
        }
    }
    // Give to the last thread the remaining work
    end = sqrt_num;

    findPrimesInRange(start, end, num, primes);

    // Wait for all the threads to finish
    pool.wait();

    // Wait for the other threads to finish
    for (thread& thread_ : threads) {
        thread_.join();
    }

    // If no prime factors found, num itself is prime
    if (primes.empty()) {
        // Again we don't need to lock the mutex as we are in the main thread
        primes.push_back({ num, 1 });
    } 
    else {
        // Check if all the prime factors have been found 
        // (otherwise a prime factor larger than the 
        // square root of the number is missing)

        unsigned long long product = 1;
        // Calculate the product of found prime factors
        for (vector<factor_exponent>::iterator it = primes.begin(); it != primes.end(); ++it) {
            // Calculate the product of the prime factors by multiplying each 
            // factor by itself for the number of times stated by the exponent
            // Not using pow() as it cuts off large numbers
            for (int i = 0; i < it->exponent; i++) {
                product *= it->factor;
            }
        }
        // If product is different from the original number, add the missing prime factor
        if (product != old_num) {
            primes.push_back({ old_num / product, 1 });
        }
    }

    // Return the vector of prime factors and their exponents
    return primes;
}

int main(int argc, char* argv[]) {
    // Check for correct number of arguments
    if (argc != 4) {
        cout << "Please provide:\n1) Number of Threads\n2) Number to be Factorized\n3) 0 or 1 to execute the program in BASH or USER mode." << endl;
        // Exit if arguments are incorrect
        return 1;
    }

    // Get the number of threads from command line arguments
    int NUM_THREADS = atoi(argv[1]);

    // Get the number to be factorized from command line arguments
    // And convert it to unsigned long long using stoull
    unsigned long long NUMBER = stoull(argv[2]);

    // Get the execution mode (BASH or USER) from command line arguments
    bool EXECUTION_MODE = atoi(argv[3]);

    // Set the affinity mask for the Main Thread
    DWORD_PTR dw = SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1));
    if (dw == 0) {
        DWORD dwErr = GetLastError();
        cerr << "SetThreadAffinityMask failed, GLE=" << dwErr << '\n';
    }

    // Creation of the Pool of Threads
    ThreadPool pool(NUM_THREADS - 1);

    // Start measuring time
    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    // Factorize the number using multiple threads
    vector<factor_exponent> factors = parallelTrialDivision(NUMBER, NUM_THREADS, pool);

    // Stop measuring time
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    // Calculate the time duration
    chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    // If USER mode
    if (EXECUTION_MODE) {
        cout << "Time taken: " << duration.count() << " milliseconds." << endl;

        cout << NUMBER << " = ";
        for (vector<factor_exponent>::iterator it = factors.begin(); it != factors.end(); ++it) {
            // Print each prime factor and its exponent
            cout << it->factor << "^" << it->exponent;
            if (next(it) != factors.end()) {
                // Print * between factors
                cout << " * ";
            }
        }
        cout << endl;
    } 
    else {
        // If BASH mode, print only the time taken
        cout << duration.count() << endl;
    }

    return 0;
}
