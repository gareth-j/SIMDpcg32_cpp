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
267816758
3390604471
4214513027
2136494172
3685676872
154162230
2002179813
1733395187
725684970
908065785
```

# Acknowledgements 
Wenzel Jakob - https://github.com/wjakob/

Daniel Lemire - https://github.com/lemire/
