#ifndef _SIMD_AVX512_PCG32_H
#define _SIMD_AVX512_PCG32_H

// An AVX-512 version of the PCG32 PRNG
// Based on code by Daniel Lemire
// https://github.com/lemire/simdpcg

// Gareth Jones - 2018

class simd_avx512_pcg32_key
{
public:
    // RNG state - 8 x 64-bits
    // All values are possible.
    __m512i state[2]; 
    // Controls which RNG sequences (stream) is selected - 8 x 64-bits
    // This must always be odd, but this is ensured by the seeding function
    // in this version 
    __m512i inc[2];

    // This can be used with both the norma and bis versions of the generator
    simd_avx512_pcg32_key()
    {
        // Get some 32-bit seeds
        std::vector<uint64_t> seed_array(32);
        std::vector<uint32_t> small_ints(64);

        randutils::auto_seed_128 seeder;

        seeder.generate(small_ints.begin(), small_ints.end());

        for(size_t i = 0; i < small_ints.size(); i+=2)
        {
            uint64_t seed_part1 = small_ints[i];
            uint64_t seed_part2 = small_ints[i+1];

            seed_array[i/2] = seed_part1 << 32 | seed_part2;
        }

        // Use some iterators to copy the values into the vectors when constructing
        // std::vector<uint64_t>::const_iterator seed_start = seed_array.begin()
        // std::vector<uint64_t>::const_iterator seed_end = seed_array.begin() + 16
        // std::vector<uint64_t>::const_iterator inc_start = seed_end + 1
        // std::vector<uint64_t> state_arr(seed_start, seed_end);
        // std::vector<uint64_t> inc_arr(inc_start, seed_array.end());

        // This is cleaner
        uint64_t state_arr[16] = 
        {
            seed_array[0], seed_array[8],
            seed_array[1], seed_array[9],
            seed_array[2], seed_array[10],
            seed_array[3], seed_array[11],
            seed_array[4], seed_array[12],
            seed_array[5], seed_array[13],
            seed_array[6], seed_array[14],
            seed_array[7], seed_array[15]
        };   


        uint64_t inc_arr[16] = 
        {
            seed_array[16], seed_array[24],
            seed_array[17], seed_array[25],
            seed_array[18], seed_array[26],
            seed_array[19], seed_array[27],
            seed_array[20], seed_array[28],
            seed_array[21], seed_array[29],
            seed_array[22], seed_array[30],
            seed_array[23], seed_array[31]
        };

        // The number one for ANDing
        const __m512i one = _mm512_set1_epi64((long long) 1);

        // Modify the seeds in init_seq to get odd values
        // Use an unaligned load here
        inc[0] = _mm512_or_si512(_mm512_slli_epi64(_mm512_loadu_si512((__m512i *) &inc_arr[0]), 1), one);
        inc[1] = _mm512_or_si512(_mm512_slli_epi64(_mm512_loadu_si512((__m512i *) &inc_arr[4]), 1), one);

        // Set the state variable with the non-deterministic seeds from randutils
        state[0] = _mm512_add_epi64(state[0], _mm512_loadu_si512((__m512i *) &state_arr[0]));
        state[1] = _mm512_add_epi64(state[1], _mm512_loadu_si512((__m512i *) &state_arr[4]));

        // We don't do the rounds of the rng here
    }     
};


class simd_avx512_pcg32
{
public:

    simd_avx512_pcg32(){} // Don't do anything here at the moment

    inline __m256i avx512_pcg32_random_r(simd_avx512_pcg32_key& key) 
    {
        const __m512i multiplier = _mm512_set1_epi64(0x5851f42d4c957f2d);
        __m512i oldstate = key.state[0];

        key.state[0] = _mm512_add_epi64(_mm512_mullo_epi64(multiplier, key.state[0]), key.inc[0]);

        __m512i xorshifted = _mm512_srli_epi64(_mm512_xor_epi64(_mm512_srli_epi64(oldstate, 18), oldstate), 27);

        __m512i rot = _mm512_srli_epi64(oldstate, 59);

        return _mm512_cvtepi64_epi32(_mm512_rorv_epi32(xorshifted, rot));
    }

    // We do the rotate using 32 bits, not the full 64 bits
    inline __m512i avx512bis_pcg32_random_r(simd_avx512_pcg32_key& key) 
    {
        const __m512i multiplier = _mm512_set1_epi64(0x5851f42d4c957f2d);

        __m512i oldstate0 = key.state[0];
        __m512i oldstate1 = key.state[1];

        __m512i lowstates = _mm512_unpacklo_epi32(oldstate1, oldstate0);
        
        key.state[0] = _mm512_add_epi64(_mm512_mullo_epi64(multiplier, key.state[0]), key.inc[0]);
        key.state[1] = _mm512_add_epi64(_mm512_mullo_epi64(multiplier, key.state[1]), key.inc[1]);

        __m512i xorshifted = _mm512_srli_epi64(_mm512_xor_epi64(_mm512_srli_epi64(lowstates, 18), lowstates), 27);

        __m512i rot = _mm512_srli_epi64(lowstates, 59);

        return _mm512_rorv_epi32(xorshifted, rot);
    }

    void populate_array_avx512_pcg32(uint32_t* rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx512_pcg32_key key;

        if (size >= 8) {
            for (; i < size - 8; i += 8) 
            {
                __m256i r = avx512_pcg32_random_r(key);
                _mm256_storeu_si256((__m256i *)(rand_arr + i), r);
            }
        }

        if (i < size) 
        {
            __m256i r = avx512_pcg32_random_r(key);
            uint32_t buffer[8];
            _mm256_storeu_si256((__m256i *)buffer, r);
            memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }
        counter += rand_arr[size - 1];
    }

    void populate_array_avx512_pcg32_two(uint32_t* rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx512_pcg32_key key1;
        simd_avx512_pcg32_key key2;

        if (size >= 16) 
        {
            for (; i < size - 16; i += 16) 
            {
                __m256i r1 = avx512_pcg32_random_r(key1);
                __m256i r2 = avx512_pcg32_random_r(key2);
                _mm256_storeu_si256((__m256i *)(rand_arr + i), r1);
                _mm256_storeu_si256((__m256i *)(rand_arr + i + 8), r2);
            }
        }

        if (size - i >= 8) 
        {
            __m256i r = avx512_pcg32_random_r(key1);
            _mm256_storeu_si256((__m256i *)(rand_arr + i), r);
            i += 8;
        }

        if (i < size) 
        {
            __m256i r = avx512_pcg32_random_r(key1);
            uint32_t buffer[8];
            _mm256_storeu_si256((__m256i *)buffer, r);
            memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];
    }

    void populate_array_avx512_pcg32_four(uint32_t* rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx512_pcg32_key key1;
        simd_avx512_pcg32_key key2;        
        simd_avx512_pcg32_key key3;
        simd_avx512_pcg32_key key4;

        if (size >= 32) 
        {
            for (; i < size - 32; i += 32) 
            {
                __m256i r1 = avx512_pcg32_random_r(key1);
                __m256i r2 = avx512_pcg32_random_r(key2);
                __m256i r3 = avx512_pcg32_random_r(key3);
                __m256i r4 = avx512_pcg32_random_r(key4);

                _mm256_storeu_si256((__m256i *)(rand_arr + i), r1);
                _mm256_storeu_si256((__m256i *)(rand_arr + i + 8), r2);
                _mm256_storeu_si256((__m256i *)(rand_arr + i + 16), r3);
                _mm256_storeu_si256((__m256i *)(rand_arr + i + 24), r4);
            }
        }

        if (size >= 16)
        {
            for (; i < size - 16; i += 16) 
            {
                __m256i r1 = avx512_pcg32_random_r(key1);
                __m256i r2 = avx512_pcg32_random_r(key2);

                _mm256_storeu_si256((__m256i *)(rand_arr + i), r1);
                _mm256_storeu_si256((__m256i *)(rand_arr + i + 8), r2);
            }
        }

        if (size - i >= 8) 
        {
            __m256i r = avx512_pcg32_random_r(key1);
            _mm256_storeu_si256((__m256i *)(rand_arr + i), r);
            i += 8;
        }

        if (i < size) 
        {
            __m256i r = avx512_pcg32_random_r(key1);
            uint32_t buffer[8];
            _mm256_storeu_si256((__m256i *)buffer, r);
            std::memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];
    }

    void populate_array_avx512bis_pcg32(uint32_t* rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx512_pcg32_key key;

        if (size >= 16) 
        {
            for (; i < size - 16; i += 16) 
            {
                __m512i r = avx512bis_pcg32_random_r(key);

                _mm512_storeu_si512((__m512i *)(rand_arr + i), r);
            }
        }

        if (i < size) 
        {
            __m512i r = avx512bis_pcg32_random_r(key);
            uint32_t buffer[16];
            _mm512_storeu_si512((__m512i *)buffer, r);
            memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];
    }

    void populate_array_avx512bis_pcg32_two(uint32_t* rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx512_pcg32_key key1;
        simd_avx512_pcg32_key key2;

        if (size >= 32) 
        {
            for (; i < size - 32; i += 32) 
            {
                __m512i r1 = avx512bis_pcg32_random_r(key1);
                __m512i r2 = avx512bis_pcg32_random_r(key2);

                _mm512_storeu_si512((__m512i *)(rand_arr + i), r1);
                _mm512_storeu_si512((__m512i *)(rand_arr + i + 16), r2);
            }
        }

        if (size >= 16) 
        {
            for (; i < size - 16; i += 16)
            {
                __m512i r = avx512bis_pcg32_random_r(key1);
                _mm512_storeu_si512((__m512i *)(rand_arr + i), r);
            }
        }

        if (i < size) 
        {
            __m512i r = avx512bis_pcg32_random_r(key1);
            uint32_t buffer[16];
            _mm512_storeu_si512((__m512i *)buffer, r);
            memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];
    }

    void populate_array_avx512bis_pcg32_four(uint32_t* rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx512_pcg32_key key1;
        simd_avx512_pcg32_key key2;        
        simd_avx512_pcg32_key key3;
        simd_avx512_pcg32_key key4;      

        if (size >= 64) 
        {
            for (; i < size - 64; i += 64) 
            {
                __m512i r1 = avx512bis_pcg32_random_r(key1);
                __m512i r2 = avx512bis_pcg32_random_r(key2);
                __m512i r3 = avx512bis_pcg32_random_r(key3);
                __m512i r4 = avx512bis_pcg32_random_r(key4);

                _mm512_storeu_si512((__m512i *)(rand_arr + i), r1);
                _mm512_storeu_si512((__m512i *)(rand_arr + i + 16), r2);
                _mm512_storeu_si512((__m512i *)(rand_arr + i + 32), r3);
                _mm512_storeu_si512((__m512i *)(rand_arr + i + 48), r4);
            }
        }

        if (size >= 32) 
        {
            for (; i < size - 32; i += 32) 
            {
                __m512i r1 = avx512bis_pcg32_random_r(key1);
                __m512i r2 = avx512bis_pcg32_random_r(key2);
                _mm512_storeu_si512((__m512i *)(rand_arr + i), r1);
                _mm512_storeu_si512((__m512i *)(rand_arr + i + 16), r2);
            }
        }

        if (size >= 16) 
        {
            for (; i < size - 16; i += 16) 
            {
                __m512i r = avx512bis_pcg32_random_r(key1);
                _mm512_storeu_si512((__m512i *)(rand_arr + i), r);
            }
        }
        if (i < size) 
        {
            __m512i r = avx512bis_pcg32_random_r(key1);
            uint32_t buffer[16];
            _mm512_storeu_si512((__m512i *)buffer, r);
            memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];
    }


    
};

#endif // __SIMD_AVX512_PCG32_H
