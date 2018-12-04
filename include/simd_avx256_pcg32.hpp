#ifndef _SIMD_AVX256_PCG32_H
#define _SIMD_AVX256_PCG32_H

// An AVX-256 version of the PCG32 PRNG
// Based on code by Daniel Lemire
// https://github.com/lemire/simdpcg

// Gareth Jones - 2018

class simd_avx256_pcg32_key
{
public:
    
    // RNG state - 4 x 64-bits
    // All values are possible.
    __m256i state; 
    
    // Controls which RNG sequences (stream) is selected - 4 x 64-bits
    // This must always be odd, but this is ensured by the seeding function
    // in this version
    __m256i inc; 

    simd_avx256_pcg32_key()
    {
        // Get some 32-bit seeds
        std::vector<uint64_t> seed_array(8);
        std::vector<uint32_t> small_ints(16);

        randutils::auto_seed_128 seeder;

        seeder.generate(small_ints.begin(), small_ints.end());

        for(size_t i = 0; i < small_ints.size(); i+=2)
        {
            uint64_t seed_part1 = small_ints[i];
            uint64_t seed_part2 = small_ints[i+1];

            seed_array[i/2] = seed_part1 << 32 | seed_part2;
        }

        uint64_t state_arr[4] = {seed_array[0],seed_array[1],seed_array[2],seed_array[3]};

        uint64_t inc_arr[4] = {seed_array[4],seed_array[5],seed_array[6],seed_array[7]};

        // The number one for ANDing
        const __m256i one = _mm256_set1_epi64x((long long) 1);

        // Set state to zero
        state = _mm256_setzero_si256();

        // Modify the seeds in init_seq to get odd values
        // inc = _mm256_or_si256(_mm256_slli_epi64(_mm256_load_si256((__m256i *) &inc_vec[0]), 1), one);

        // Use an unaligned load here
        inc = _mm256_or_si256(_mm256_slli_epi64(_mm256_loadu_si256((__m256i *) &inc_arr[0]), 1), one);

        
        // Set the state variable with the non-deterministic seeds from randutils
        state = _mm256_add_epi64(state, _mm256_loadu_si256((__m256i *) &state_arr[0]));

        // We don't do the rounds of the rng here but it should be OK as the seeds are from
        // a decent source?

    }
};


class simd_avx256_pcg32
{
private:
    // untested
    inline __m256i hacked_mm256_rorv_epi32(__m256i x, __m256i r) 
    {   
        // Subtract 32 from r
        __m256i subtracted = _mm256_sub_epi32(_mm256_set1_epi32(32), r);

        // Shift x left by subtracted
        __m256i left_shifted = _mm256_sllv_epi32(x, subtracted);

        // Shift x right by r
        __m256i right_shifted = _mm256_srlv_epi32(x, r);
        
        // Bitwise OR of integer data between a and b
        return _mm256_or_si256(left_shifted, right_shifted);
    }

    // untested
    inline __m256i hacked_mm256_mullo_epi64(__m256i x, __m256i ml, __m256i mh) 
    {
        // Bitwise AND or x and the 64-bit int
        __m256i xl = _mm256_and_si256(x, _mm256_set1_epi64x(UINT64_C(0x00000000ffffffff)));
        // Shift x left by 32 bits
        __m256i xh = _mm256_srli_epi64(x, 32);

        // Multiply low 32-bit integers of xh and ml and shift the result by 32 
        __m256i hl = _mm256_slli_epi64(_mm256_mul_epu32(xh, ml), 32);
        // Same but with xl and mh
        __m256i lh = _mm256_slli_epi64(_mm256_mul_epu32(xl, mh), 32);
        // Multiply the low 32-bit integers of xl and ml
        __m256i ll = _mm256_mul_epu32(xl, ml);

        // Add ll and the result of adding hl and lh
        return _mm256_add_epi64(ll, _mm256_add_epi64(hl, lh));
    }

    inline __m128i avx256_pcg32_random_r(simd_avx256_pcg32_key& key) 
    {
        // Multiplication constant
        const unsigned long long PCG32_MULT = 0x5851f42d4c957f2d;
        const __m256i pcg32_mult_l = _mm256_set1_epi64x((long long) (PCG32_MULT & 0xffffffffu));
        const __m256i pcg32_mult_h = _mm256_set1_epi64x((long long) (PCG32_MULT >> 32));

        __m256i oldstate = key.state;

        key.state = _mm256_add_epi64(hacked_mm256_mullo_epi64(key.state, pcg32_mult_l, pcg32_mult_h), key.inc);

        __m256i xorshifted = _mm256_srli_epi64(_mm256_xor_si256(_mm256_srli_epi64(oldstate, 18), oldstate), 27);
        
        __m256i rot = _mm256_srli_epi64(oldstate, 59);
        
        return _mm256_castsi256_si128(
                   _mm256_permutevar8x32_epi32(hacked_mm256_rorv_epi32(xorshifted, rot),
                                               _mm256_set_epi32(7, 7, 7, 7, 6, 4, 2, 0)));
    }  

public:

    simd_avx256_pcg32(){} // Do nothing here at the moment    

    void populate_array_avx256_pcg32(uint32_t* rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx256_pcg32_key key;

        if (size >= 4) 
        {
            for (; i < size - 4; i += 4) 
            {
                __m128i r = avx256_pcg32_random_r(key);
                _mm_storeu_si128((__m128i *)(rand_arr + i), r);
            }
        }

        if (i < size) 
        {
            __m128i r = avx256_pcg32_random_r(key);
            uint32_t buffer[4];
            _mm_storeu_si128((__m128i *)buffer, r);
            std::memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];
    }

    void populate_array_avx256_pcg32_two(uint32_t* rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx256_pcg32_key key1;
        simd_avx256_pcg32_key key2;


        if (size >= 8) 
        {
            for (; i < size - 8; i += 8) 
            {
                __m128i r1 = avx256_pcg32_random_r(key1);
                __m128i r2 = avx256_pcg32_random_r(key2);

                _mm_storeu_si128((__m128i *)(rand_arr + i), r1);
                _mm_storeu_si128((__m128i *)(rand_arr + i + 4), r2);
            }
        }

        if (size - i >= 4) 
        {
            __m128i r = avx256_pcg32_random_r(key1);
            _mm_storeu_si128((__m128i *)(rand_arr + i), r);
            i += 4;
        }

        if (i < size) 
        {
            __m128i r = avx256_pcg32_random_r(key1);
            uint32_t buffer[8];
            _mm_storeu_si128((__m128i *)buffer, r);
            std::memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];    
    }

    void populate_array_avx256_pcg32_four(uint32_t* rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        // Get four automatically seeded key objects
        simd_avx256_pcg32_key key1;            
        simd_avx256_pcg32_key key2;
        simd_avx256_pcg32_key key3;
        simd_avx256_pcg32_key key4;

        if (size >= 16) 
        {
            for (; i < size - 16; i += 16) 
            {
                __m128i r1 = avx256_pcg32_random_r(key1);
                __m128i r2 = avx256_pcg32_random_r(key2);
                __m128i r3 = avx256_pcg32_random_r(key3);
                __m128i r4 = avx256_pcg32_random_r(key4);

                _mm_storeu_si128((__m128i *)(rand_arr + i), r1);
                _mm_storeu_si128((__m128i *)(rand_arr + i + 4), r2);
                _mm_storeu_si128((__m128i *)(rand_arr + i + 8), r3);
                _mm_storeu_si128((__m128i *)(rand_arr + i + 12), r4);
            }
        }
        
        if (size >= 8) 
        {
            for (; i < size - 8; i += 8) 
            {
                __m128i r1 = avx256_pcg32_random_r(key1);
                __m128i r2 = avx256_pcg32_random_r(key2);
            
                _mm_storeu_si128((__m128i *)(rand_arr + i), r1);
                _mm_storeu_si128((__m128i *)(rand_arr + i + 4), r2);
            }
        }

        if (size - i >= 4) 
        {
            __m128i r = avx256_pcg32_random_r(key1);
            _mm_storeu_si128((__m128i *)(rand_arr + i), r);
            i += 8;
        }

        if (i < size) 
        {
            __m128i r = avx256_pcg32_random_r(key1);
            uint32_t buffer[4];
            _mm_storeu_si128((__m128i *)buffer, r);
            memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];
    }
};


#endif // _SIMD_AVX256_PCG32_H