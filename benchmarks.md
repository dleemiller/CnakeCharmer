# Benchmark Report

| Benchmark | Variant | Python Avg (s) | Python Std (s) | Cython Avg (s) | Cython Std (s) | Speedup |
|-----------|-----------|----------------|----------------|----------------|----------------|---------|
| fib | cython | 0.000009 | 0.000000 | 0.000003 | 0.000000 | 2.88x |
| primes | cython | 0.001743 | 0.000226 | 0.000131 | 0.000001 | 13.28x |
| fizzbuzz | cython | 0.001201 | 0.000075 | 0.000366 | 0.000002 | 3.28x |
| fizzbuzz | pure py | 0.001201 | 0.000075 | 0.000392 | 0.000072 | 3.06x |
