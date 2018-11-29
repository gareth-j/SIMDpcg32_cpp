/*
 * Vectorized AVX2 version of the PCG32 random number generator developed by
 * Wenzel Jakob (June 2016)
 *
 * The PCG random number generator was developed by Melissa O'Neill
 * <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

#ifndef _PCG32_8_H
#define _PCG32_8_H

#include <utility>
#include <immintrin.h>
#include <cstring>
#include "pcg32.hpp"

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

// 8 parallel PCG32 pseudorandom number generators
class PCG32_ALIGN(32) pcg32_8 
{
protected:
	// Multiplication constant
	const unsigned long long PCG32_MULT = 0x5851f42d4c957f2d;

public:

#if defined(__AVX2__)
	// RNG state.  All values are possible.
    __m256i state[2];
    // Controls which RNG sequence (stream) is selected. Must *always* be odd.
    // The seed() function will ensure that the value of inc is odd 
    __m256i inc[2];   
#else
    // If we haven't got AVX2, just use the standard rng
    std::array<pcg32, 8> rng;
#endif

    static const size_t N_seeds = 16;
    static const size_t N_seed_part = 32;
    
    // Can do it one at a time
    uint64_t get_state_seed()
    {
    	std::array<uint32_t, 2> seed_array;

	    randutils::auto_seed_128 seeder;

		seeder.generate(seed_array.begin(), seed_array.end());

		uint64_t seed_part1 = seed_array[0];
		uint64_t seed_part2 = seed_array[1];
		
		// Create some 64-bit numbers
		uint64_t seed = seed_part1 << 32 | seed_part2;

		return seed;
    }

    // Or create an array of 64-bit ints
    std::array<uint64_t, N_seeds> get_seed_array()
    {
    	std::array<uint32_t, N_seed_part> small_ints;
    	std::array<uint64_t, N_seeds> seed_array;

	    randutils::auto_seed_128 seeder;

		seeder.generate(small_ints.begin(), small_ints.end());

		for(size_t i = 0; i < N_seed_part; i+=2)
		{
			uint64_t seed_part1 = small_ints[i];
			uint64_t seed_part2 = small_ints[i+1];

			seed_array[i/2] = seed_part1 << 32 | seed_part2;
		}		

		return seed_array;
    }

	// Initialize the pseudorandom number generator with a non-deterministic seed
    pcg32_8() 
    {
    	std::array<uint64_t, N_seeds> seed_array = get_seed_array();

        PCG32_ALIGN(32) uint64_t init_state[8] = 
        {
			seed_array[0], seed_array[4],
			seed_array[1], seed_array[5],
			seed_array[2], seed_array[6],
			seed_array[3], seed_array[7]
        };   

        PCG32_ALIGN(32) uint64_t init_seq[8] = 
        {
			seed_array[8], seed_array[12],
			seed_array[9], seed_array[13],
			seed_array[10], seed_array[14],
			seed_array[11], seed_array[15]
		};

		// Same seeding process used in O'Neil's C code
		set_seeds(init_state, init_seq);
    }



#if defined(__AVX2__)

    // Fill the array rand_arr with pseudo-random numbers
    void populate_array_avx2_pcg32(uint32_t* rand_arr, const uint32_t size) 
	{
		uint32_t i = 0;

		// Don't need the key as the class is already seeded
		if (size >= 8) 
		{
			for(; i < size - 8; i += 8) 
			{
				__m256i r = get_rand();
				_mm256_storeu_si256((__m256i *)(rand_arr + i), r);		    	
			}
		}

		if (i < size) 
		{
			__m256i r = get_rand();
			uint32_t buffer[8];
			_mm256_storeu_si256((__m256i *)buffer, r);
			std::memcpy(rand_arr + i, buffer, sizeof(uint32_t) * (size - i));
		}

		counter += rand_arr[size - 1];
	}
    
    void set_seeds(const uint64_t init_state[8], const uint64_t init_seq[8]) 
    {
        const __m256i one = _mm256_set1_epi64x((long long) 1);

        // Set to be zeroes
        state[0] = state[1] = _mm256_setzero_si256();

        // Modify the seeds in init_seq to get odd values
        inc[0] = _mm256_or_si256(_mm256_slli_epi64(_mm256_load_si256((__m256i *) &init_seq[0]), 1), one);
        inc[1] = _mm256_or_si256(_mm256_slli_epi64(_mm256_load_si256((__m256i *) &init_seq[4]), 1), one);
        
        // Call the RNG
        step();

        // Add the seed variables from init_state to the current state variables
        state[0] = _mm256_add_epi64(state[0], _mm256_load_si256((__m256i *) &init_state[0]));
        state[1] = _mm256_add_epi64(state[1], _mm256_load_si256((__m256i *) &init_state[4]));

        // Call the RNG again
        step();
    }

    // Generate 8 uniformly distributed unsigned 32-bit random numbers
    // and store them in result
    void get_rand(uint32_t result[8]) 
    {
        _mm256_store_si256((__m256i *) result, step());
    }

    // Generate 8 uniformly distributed unsigned 32-bit random numbers
    __m256i PCG32_VECTORCALL get_rand() 
    {
        return step();
    }

    // Generate eight single precision floating point value on the interval [0, 1)
    __m256 PCG32_VECTORCALL get_float() 
    {
        /* Trick from MTGP: generate an uniformly distributed
           single precision number in [1,2) and subtract 1. */

        const __m256i const1 = _mm256_set1_epi32((int) 0x3f800000u);

        __m256i value = step();
        __m256i fltval = _mm256_or_si256(_mm256_srli_epi32(value, 9), const1);

        return _mm256_sub_ps(_mm256_castsi256_ps(fltval),
                             _mm256_castsi256_ps(const1));
    }

    // Generate eight single precision floating point value on the interval [0, 1)
    void get_float(float result[8]) 
    {
        _mm256_store_ps(result, get_float());
    }

    
    // This returns a pair of 256-bit variables, each of which represents 4 doubles
    std::pair<__m256d, __m256d> get_double() 
    {
		// Trick from MTGP: generate an uniformly distributed
		// double precision number in [1,2) and subtract 1.

        const __m256i const1 = _mm256_set1_epi64x((long long) 0x3ff0000000000000ull);

        __m256i value = step();

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
    void get_double(double result[8])
    {
        std::pair<__m256d, __m256d> value = get_double();

        _mm256_store_pd(&result[0], value.first);
        _mm256_store_pd(&result[4], value.second);
    }

private:

    PCG32_INLINE __m256i PCG32_VECTORCALL step()
    {
        const __m256i pcg32_mult_l = _mm256_set1_epi64x((long long) (PCG32_MULT & 0xffffffffu));
        const __m256i pcg32_mult_h = _mm256_set1_epi64x((long long) (PCG32_MULT >> 32));
        const __m256i mask_l       = _mm256_set1_epi64x((long long) 0x00000000ffffffffull);
        const __m256i shift0       = _mm256_set_epi32(7, 7, 7, 7, 6, 4, 2, 0);
        const __m256i shift1       = _mm256_set_epi32(6, 4, 2, 0, 7, 7, 7, 7);
        const __m256i const32      = _mm256_set1_epi32(32);

        __m256i s0 = state[0], s1 = state[1];

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

        state[0] = _mm256_add_epi64(s0n, inc[0]);
        state[1] = _mm256_add_epi64(s1n, inc[1]);

        /* Finally, rotate and return the result */
        __m256i result = _mm256_or_si256(
            _mm256_srlv_epi32(xors, rot),
            _mm256_sllv_epi32(xors, _mm256_sub_epi32(const32, rot))
        );

        return result;
    }

// If we haven't got AVX2
#else 

    void populate_array_avx2_pcg32(uint32_t* rand_arr, const uint32_t size)
    {
    	uint32_t i = 0;

    	// We have 8 generators, use them to fill the
    	uint32_t total_size = size;
    	
    	int n_generators = 8;
    	
    	uint32_t blocks_per_generator = total_size/n_generators;

    	uint32_t leftover = total_size - (blocks_per_generator * n_generators);

	   	#pragma omp parallel
	   	{
			int thread_no = omp_get_thread_num();
			uint32_t start_block = thread_no * blocks_per_generator;
			uint32_t end_block = start_block + blocks_per_generator;

			#pragma omp for
			for(uint32_t i = start_block; i < end_block; ++i)
	    	{
	    		rand_arr[i] = get_rand();
	    	}
    	}

    	if(leftover > 0)
    	{
    		for(uint32_t i = (size - leftover); i < leftover; ++i)
    		{
    			rand_arr[i] = get_rand();
    		}
    	}
    }

    // If we haven't got AVX just used the normal version on arrays of 8 rands at a time
    // They don't need seeding as they're seeding automatically

    // Generate 8 uniformly distributed unsigned 32-bit random numbers
    void get_rand(uint32_t result[8]) 
    {
        for (int i = 0; i < 8; ++i)
            result[i] = rng[i].get_rand();
    }

    // Generate eight single precision floating point value on the interval [0, 1)
    void get_float(float result[8]) 
    {
        for (int i = 0; i < 8; ++i)
            result[i] = rng[i].get_float();
    }

	// Generate eight double precision floating point value on the interval [0, 1)
	void get_double(double result[8]) 
	{
		for (int i = 0; i < 8; ++i)
		    result[i] = rng[i].get_double();
	}

#endif

};


#endif // __PCG32_8_H
