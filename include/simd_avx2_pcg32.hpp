#ifndef _SIMD_AVX2_PCG32_H
#define _SIMD_AVX2_PCG32_H

#include <utility>
#include <immintrin.h>
#include <cstring>
#include "pcg32.hpp"

// An AVX-2 version of the PCG32 PRNG
// Based on code by Daniel Lemire
// https://github.com/lemire/simdpcg
// Credit Wenzel Jakob
// https://github.com/wjakob/pcg32/blob/master/pcg32_8.h

// Gareth Jones - 2018

#if defined(_MSC_VER)
#  define PCG32_ALIGN(amt)    __declspec(align(amt))
#  define PCG32_VECTORCALL    __vectorcall
#  define PCG32_INLINE        __forceinline
#else

// Align the array on a 32-byte boundary
#  define PCG32_ALIGN(amt)    __attribute__ ((aligned(amt)))
#  define PCG32_INLINE        __attribute__ ((always_inline))
#  if defined(__clang__)
#    define PCG32_VECTORCALL  __attribute__ ((vectorcall))
#  else
#    define PCG32_VECTORCALL
#  endif
#endif

static uint32_t counter;

class simd_avx2_pcg32_key
{
public:
    
    // RNG state - 8 x 64-bits
    // All values are possible.
    __m256i state[2]; 
    
    // Controls which RNG sequences (stream) is selected - 8 x 64-bits
    // This must always be odd, but this is ensured by the seeding function
    // in this version
    __m256i inc[2]; 

    simd_avx2_pcg32_key()
    {
        // Get some 32-bit seeds
        std::vector<uint64_t> seed_array(16);
        std::vector<uint32_t> small_ints(32);

        randutils::auto_seed_128 seeder;

        seeder.generate(small_ints.begin(), small_ints.end());

        for(size_t i = 0; i < small_ints.size(); i+=2)
        {
            uint64_t seed_part1 = small_ints[i];
            uint64_t seed_part2 = small_ints[i+1];

            seed_array[i/2] = seed_part1 << 32 | seed_part2;
        }

        uint64_t state_arr[8] = 
        {
            seed_array[0], seed_array[4],
            seed_array[1], seed_array[5],
            seed_array[2], seed_array[6],
            seed_array[3], seed_array[7]
        };   

        uint64_t inc_arr[8] = 
        {
            seed_array[8], seed_array[12],
            seed_array[9], seed_array[13],
            seed_array[10], seed_array[14],
            seed_array[11], seed_array[15]
        };

        const __m256i one = _mm256_set1_epi64x((long long) 1);

        // Set to be zeroes
        state[0] = state[1] = _mm256_setzero_si256();

        // Modify the seeds in init_seq to get odd values
        inc[0] = _mm256_or_si256(_mm256_slli_epi64(_mm256_load_si256((__m256i *) &inc_arr[0]), 1), one);
        inc[1] = _mm256_or_si256(_mm256_slli_epi64(_mm256_load_si256((__m256i *) &inc_arr[4]), 1), one);

        // Call the RNG
        // avx2_pcg32_random_r();
        
        // Add the seed variables from init_state to the current state variables
        state[0] = _mm256_add_epi64(state[0], _mm256_load_si256((__m256i *) &state_arr[0]));
        state[1] = _mm256_add_epi64(state[1], _mm256_load_si256((__m256i *) &state_arr[4]));

        // // Call the RNG again
        // avx2_pcg32_random_r();
        // We don't do the rounds of the RNG but hopefully the seeds are good enough
    }
};

// The AVX2 version of the generator
class simd_avx2_pcg32 
{
public:

    simd_avx2_pcg32(){} // Do nothing here at the moment

    // Fill the array rand_arr with pseudo-random numbers
    void populate_array_simd_pcg32(uint32_t* rand_arr, const uint32_t size) 
	{
		uint32_t i = 0;

        // Create a seeded key object
        simd_avx2_pcg32_key key;

		// Don't need the key as the class is already seeded
		if (size >= 8) 
		{
			for(; i < size - 8; i += 8) 
			{
				__m256i r = get_rand(key);
				_mm256_storeu_si256((__m256i *)(rand_arr + i), r);		    	
			}
		}

		if (i < size) 
		{
			__m256i r = get_rand(key);
			uint32_t buffer[8];
			_mm256_storeu_si256((__m256i *)buffer, r);
			std::memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
		}

		counter += rand_arr[size - 1];
	}

    void populate_array_simd_pcg32_two(uint32_t *rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx2_pcg32_key key1;
        simd_avx2_pcg32_key key2;

        if (size >= 16) 
        {
            for (; i < size - 16; i += 16) 
            {
                __m256i r1 = avx2_pcg32_random_r(key1);
                __m256i r2 = avx2_pcg32_random_r(key2);

                _mm256_storeu_si256((__m256i *)(rand_arr + i), r1);
                _mm256_storeu_si256((__m256i *)(rand_arr + i + 8), r2);
            }
        }

        if (size - i >= 8) 
        {
            __m256i r = avx2_pcg32_random_r(key1);
           
            _mm256_storeu_si256((__m256i *)(rand_arr + i), r);
            
            i += 8;
        }
            
        if (i < size) 
        {
            __m256i r = avx2_pcg32_random_r(key1);

            uint32_t buffer[8];

            _mm256_storeu_si256((__m256i *)buffer, r);

            std::memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];
    }

    void populate_array_simd_pcg32_four(uint32_t *rand_arr, const uint32_t size) 
    {
        uint32_t i = 0;

        simd_avx2_pcg32_key key1;
        simd_avx2_pcg32_key key2;
        simd_avx2_pcg32_key key3;
        simd_avx2_pcg32_key key4;        

        if (size >= 32) 
        {
            for (; i < size - 32; i += 32) 
            {
                __m256i r1 = avx2_pcg32_random_r(key1);
                __m256i r2 = avx2_pcg32_random_r(key2);
                __m256i r3 = avx2_pcg32_random_r(key3);
                __m256i r4 = avx2_pcg32_random_r(key4);

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
                __m256i r1 = avx2_pcg32_random_r(key1);
                __m256i r2 = avx2_pcg32_random_r(key2);
                
                _mm256_storeu_si256((__m256i *)(rand_arr + i), r1);
                _mm256_storeu_si256((__m256i *)(rand_arr + i + 8), r2);
            }
        }

        if (size - i >= 8) 
        {
            __m256i r = avx2_pcg32_random_r(key1);
            _mm256_storeu_si256((__m256i *)(rand_arr + i), r);
            i += 8;
        }

        if (i < size) 
        {
            __m256i r = avx2_pcg32_random_r(key1);
            
            uint32_t buffer[8];

            _mm256_storeu_si256((__m256i *)buffer, r);
            
            std::memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
        }

        counter += rand_arr[size - 1];
    }
   
    // Generate 8 uniformly distributed unsigned 32-bit random numbers
    // and store them in result
    void get_rand(uint32_t result[8], simd_avx2_pcg32_key& key) 
    {
        _mm256_store_si256((__m256i *) result, avx2_pcg32_random_r(key));
    }

    // Generate 8 uniformly distributed unsigned 32-bit random numbers
    __m256i PCG32_VECTORCALL get_rand(simd_avx2_pcg32_key& key) 
    {
        return avx2_pcg32_random_r(key);
    }

    // Generate eight single precision floating point value on the interval [0, 1)
    __m256 PCG32_VECTORCALL get_float(simd_avx2_pcg32_key& key) 
    {
        /* Trick from MTGP: generate an uniformly distributed
           single precision number in [1,2) and subtract 1. */

        const __m256i const1 = _mm256_set1_epi32((int) 0x3f800000u);

        __m256i value = avx2_pcg32_random_r(key);
        __m256i fltval = _mm256_or_si256(_mm256_srli_epi32(value, 9), const1);

        return _mm256_sub_ps(_mm256_castsi256_ps(fltval),
                             _mm256_castsi256_ps(const1));
    }

    // Generate eight single precision floating point value on the interval [0, 1)
    void get_float(float result[8], simd_avx2_pcg32_key& key) 
    {
        _mm256_store_ps(result, get_float(key));
    }

  
    // This returns a pair of 256-bit variables, each of which represents 4 doubles
    std::pair<__m256d, __m256d> get_double(simd_avx2_pcg32_key& key) 
    {
		// Trick from MTGP: generate an uniformly distributed
		// double precision number in [1,2) and subtract 1.

        const __m256i const1 = _mm256_set1_epi64x((long long) 0x3ff0000000000000ull);

        __m256i value = avx2_pcg32_random_r(key);

        __m256i lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(value));
        __m256i hi = _mm256_cvtepu32_epi64(_mm256_extractf128_si256(value, 1));

        __m256i tlo = _mm256_or_si256(_mm256_slli_epi64(lo, 20), const1);
        __m256i thi = _mm256_or_si256(_mm256_slli_epi64(hi, 20), const1);

        __m256d flo = _mm256_sub_pd(_mm256_castsi256_pd(tlo),
                                    _mm256_castsi256_pd(const1));

        __m256d fhi = _mm256_sub_pd(_mm256_castsi256_pd(thi),
                                    _mm256_castsi256_pd(const1));

        return std::make_pair(flo, fhi);
    }

    
    // Generate eight double precision floating point value on the interval [0, 1)
    // Only have 32 bits of mantissa here but greater precision than float that has 23 bits
    // Fills an array with 8 doubles
    void get_double(double result[8], simd_avx2_pcg32_key& key)
    {
        std::pair<__m256d, __m256d> value = get_double(key);

        _mm256_store_pd(&result[0], value.first);
        _mm256_store_pd(&result[4], value.second);
    }

private:

    PCG32_INLINE __m256i PCG32_VECTORCALL avx2_pcg32_random_r(simd_avx2_pcg32_key& key)
    {
            // Multiplication constant
        const unsigned long long PCG32_MULT = 0x5851f42d4c957f2d;
        const __m256i pcg32_mult_l = _mm256_set1_epi64x((long long) (PCG32_MULT & 0xffffffffu));
        const __m256i pcg32_mult_h = _mm256_set1_epi64x((long long) (PCG32_MULT >> 32));
        const __m256i mask_l       = _mm256_set1_epi64x((long long) 0x00000000ffffffffull);
        const __m256i shift0       = _mm256_set_epi32(7, 7, 7, 7, 6, 4, 2, 0);
        const __m256i shift1       = _mm256_set_epi32(6, 4, 2, 0, 7, 7, 7, 7);
        const __m256i const32      = _mm256_set1_epi32(32);

        __m256i s0 = key.state[0], s1 = key.state[1];

        /* Extract low and high words for partial products below */
        __m256i s0_l = _mm256_and_si256(s0, mask_l);
        __m256i s0_h = _mm256_srli_epi64(s0, 32);
        __m256i s1_l = _mm256_and_si256(s1, mask_l);
        __m256i s1_h = _mm256_srli_epi64(s1, 32);

        /* Improve high bits using xorshift step */
        __m256i s0s   = _mm256_srli_epi64(s0, 18);
        __m256i s1s   = _mm256_srli_epi64(s1, 18);

        __m256i s0x   = _mm256_xor_si256(s0s, s0);
        __m256i s1x   = _mm256_xor_si256(s1s, s1);

        __m256i s0xs  = _mm256_srli_epi64(s0x, 27);
        __m256i s1xs  = _mm256_srli_epi64(s1x, 27);

        __m256i xors0 = _mm256_and_si256(mask_l, s0xs);
        __m256i xors1 = _mm256_and_si256(mask_l, s1xs);

        /* Use high bits to choose a bit-level rotation */
        __m256i rot0  = _mm256_srli_epi64(s0, 59);
        __m256i rot1  = _mm256_srli_epi64(s1, 59);

        /* 64 bit multiplication using 32 bit partial products :( */
        __m256i m0_hl = _mm256_mul_epu32(s0_h, pcg32_mult_l);
        __m256i m1_hl = _mm256_mul_epu32(s1_h, pcg32_mult_l);
        __m256i m0_lh = _mm256_mul_epu32(s0_l, pcg32_mult_h);
        __m256i m1_lh = _mm256_mul_epu32(s1_l, pcg32_mult_h);

        /* Assemble lower 32 bits, will be merged into one 256 bit vector below */
        xors0 = _mm256_permutevar8x32_epi32(xors0, shift0);
        rot0  = _mm256_permutevar8x32_epi32(rot0, shift0);
        xors1 = _mm256_permutevar8x32_epi32(xors1, shift1);
        rot1  = _mm256_permutevar8x32_epi32(rot1, shift1);

        /* Continue with partial products */
        __m256i m0_ll = _mm256_mul_epu32(s0_l, pcg32_mult_l);
        __m256i m1_ll = _mm256_mul_epu32(s1_l, pcg32_mult_l);

        __m256i m0h   = _mm256_add_epi64(m0_hl, m0_lh);
        __m256i m1h   = _mm256_add_epi64(m1_hl, m1_lh);

        __m256i m0hs  = _mm256_slli_epi64(m0h, 32);
        __m256i m1hs  = _mm256_slli_epi64(m1h, 32);

        __m256i s0n   = _mm256_add_epi64(m0hs, m0_ll);
        __m256i s1n   = _mm256_add_epi64(m1hs, m1_ll);

        __m256i xors  = _mm256_or_si256(xors0, xors1);
        __m256i rot   = _mm256_or_si256(rot0, rot1);

        key.state[0] = _mm256_add_epi64(s0n, key.inc[0]);
        key.state[1] = _mm256_add_epi64(s1n, key.inc[1]);

        /* Finally, rotate and return the result */
        __m256i result = _mm256_or_si256(
            _mm256_srlv_epi32(xors, rot),
            _mm256_sllv_epi32(xors, _mm256_sub_epi32(const32, rot))
        );

        return result;
    }

};




#endif // _SIMD_AVX2_PCG32_H
