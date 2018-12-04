# SIMDpcg32_cpp

A vectorized C++ implementation of the 32-bit PCG pseudo-random number generator by 
Melissa O'Neill available at http://www.pcg-random.org.

This version is based on the work of Wenzel Jakob at https://github.com/wjakob/pcg32

I've added some improved seeding functions. Will add AVX-512 implementation by
Daniel Lemire and benchmarking code next.

## Use
```
make
./simd_pcg
```
And you should see 
```
Some random numbers...
2478751624
3133991757
3414538778
3791279692
470550470
3188559825
3585534050
3657948795
1651540920
3305040742

==========================
	Generators
==========================

Testing function : pcg32
3.1 cycles per operation
Testing function : populate_array_simd_pcg32
1.6 cycles per operation
Testing function : populate_array_simd_pcg32_two
1.4 cycles per operation
Testing function : populate_array_simd_pcg32_four
1.3 cycles per operation
Testing function : populate_array_avx256_pcg32
2.1 cycles per operation
Testing function : populate_array_avx256_pcg32_two
1.6 cycles per operation
Testing function : populate_array_avx256_pcg32_two
1.6 cycles per operation
Testing function : populate_array_avx512_pcg32
2 cycles per operation
Testing function : populate_array_avx512_pcg32_two
1.3 cycles per operation
Testing function : populate_array_avx512_pcg32_four
0.93 cycles per operation
Testing function : populate_array_avx512bis_pcg32
1 cycles per operation
Testing function : populate_array_avx512bis_pcg32_two
0.83 cycles per operation
Testing function : populate_array_avx512bis_pcg32_four
0.64 cycles per operation
```

# Acknowledgements 
Wenzel Jakob - https://github.com/wjakob/

Daniel Lemire - https://github.com/lemire/
