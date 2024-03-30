Consigli per testare l'Algoritmo scelto:

- Variare il numero di Thread: dopo il numero di Thread logici disponibili sul nostro processore è inutile andare avanti di molto, perché l'overhead introdotto supera il beneficio che ne deriva.
- Variare il carico: il punto di saturazione si raggiunge quando le prestazioni non migliorano più.

Studiare la scalabilità dell'Algoritmo e presentare al Professore risultati concreti con grafici di facile lettura.

---

Da fare:

- Aggiungere controllo argc se è maggiore di due
- Vedere la modifica con struttura con numero ed esponente e modificare findPrimeInRange
- Per futuri grafici mettere più thread per vedere che non migliora (hardware + hardware/2)
- Fare ottimizzazione range: sqrt(N) - 3

--- 

Things to insert in the documentation:

Chosen function for measuring the execution time. Both `clock()` in C and `chrono::high_resolution_clock::now()` in C++ are commonly used for measuring the duration of a program or a specific section of code. However, there are some differences between them:

1. **Resolution**:
   - `clock()`: The resolution of `clock()` is implementation-dependent and may not always be high. It typically measures time in clock ticks.
   - `chrono::high_resolution_clock::now()`: This provides the highest resolution time available on the system. It is generally more precise than `clock()`.

2. **Portability**:
   - `clock()`: It's part of the C standard library and is thus more portable across different platforms and compilers.
   - `chrono::high_resolution_clock::now()`: It's part of the C++ `<chrono>` library, which may not be available in all C++ implementations.

3. **Syntax**:
   - `clock()`: It's a C function, so it's used in C programs.
   - `chrono::high_resolution_clock::now()`: It's used in C++ programs.

If you're working in C, and portability across different platforms is important, `clock()` is a reasonable choice. However, if you're working in C++ and need higher resolution timing, `chrono::high_resolution_clock::now()` is generally preferred.

Additionally, in C++, you can use `std::chrono` library for more functionality and flexibility in time measurements. For example, you can measure durations more accurately and with more control using `std::chrono::steady_clock` or `std::chrono::system_clock`.

Here's an example of how you might use `std::chrono::high_resolution_clock::now()` in C++:

```cpp
#include <iostream>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // Your code to measure here

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}
```

Ultimately, the choice between `clock()` in C and `chrono::high_resolution_clock::now()` in C++ depends on your specific requirements, the platform you're targeting, and your personal preferences.


---

References

- https://github.com/Albran99/BitonicSort/tree/main
- https://github.com/FedericoCavedoni/ComputerArchitecture_Project/tree/main