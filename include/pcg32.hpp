// Tiny self-contained version of the PCG Random Number Generation for C++
// put together from pieces of the much larger C/C++ codebase.
// Wenzel Jakob, February 2015
// 
// The PCG random number generator was developed by Melissa O'Neill
// <oneill@pcg-random.org>
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// For additional information about the PCG random number generation scheme,
// including its license and other licensing options, visit
// 
//     http://www.pcg-random.org
 
#ifndef _PCG32_H
#define _PCG32_H

// #define PCG32_MULT 0x5851f42d4c957f2dULL

#include <cstdint>
#include <cmath>
#include <cassert>
#include <algorithm>

#include "randutils.hpp"

/// PCG32 Pseudorandom number generator

class pcg32
{
public:
    // Initialize the pseudorandom number generator with a non-deterministic
    // seeds from randutils
    pcg32()
    {
    	std::array<uint32_t, 4> seed_array;

	    randutils::auto_seed_128 seeder;
		
		seeder.generate(seed_array.begin(), seed_array.end());

		uint64_t seed_part1 = seed_array[0];
		uint64_t seed_part2 = seed_array[1];
		uint64_t seed_part3 = seed_array[2];
		uint64_t seed_part4 = seed_array[3];

		// Create some 64-bit numbers
		uint64_t seed1 = seed_part1 << 32 | seed_part2;
		uint64_t seed2 = seed_part3 << 32 | seed_part4;

		// The same seeding process used in the O'Neill's C code
		// This ensures inc is odd
		state = 0U;
		inc = (seed2 << 1u) | 1u;
		get_rand();
		state += seed1;
		get_rand();
    }

    // Generate a uniformly distributed unsigned 32-bit random number
    uint32_t get_rand() 
    {
        uint64_t oldstate = state;

        state = oldstate * PCG32_MULT + inc;

        uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);

        uint32_t rot = (uint32_t) (oldstate >> 59u);

        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

    /// Generate a uniformly distributed number, r, where 0 <= r < bound
    uint32_t get_bounded_rand(uint32_t bound) 
    {
    	// See Notes 1.0
        uint32_t threshold = (~bound+1u) % bound;

        // Uniformity should guarantee this terminates, usually quickly.
        // See Notes 1.1
        for (;;) 
        {
            uint32_t r = get_rand();
            if (r >= threshold)
                return r % bound;
        }
    }

	// Generate a single precision floating point value on the interval [0, 1)
    float get_float() 
    {
		// Trick from MTGP: generate an uniformly distributed
		// single precision number in [1,2) and subtract 1.
        union 
        {
            uint32_t u;
            float f;
        } x;

        x.u = (get_rand() >> 9) | 0x3f800000u;
        return x.f - 1.0f;
    }

    // Only 32 bits of the mantissa will be filled for this double
    // Still greater precision than using float which uses 23 bits
    // Interval [0, 1)
    double get_double() 
    {
        // Trick from MTGP: generate an uniformly distributed
        //  double precision number in [1,2) and subtract 1. 
    	// where MTGP - Mersenne Twister Graphic Processing
        union 
        {
            uint64_t u;
            double d;
        } x;

        x.u = ((uint64_t) get_rand() << 20) | 0x3ff0000000000000ULL;
        return x.d - 1.0;
    }

    
    // Multi-step advance function (jump-ahead, jump-back)
    // See Notes 1.2
    void advance(int64_t delta_) 
    {
        uint64_t
            cur_mult = PCG32_MULT,
            cur_plus = inc,
            acc_mult = 1u,
            acc_plus = 0u;

		// Even though delta is an unsigned integer, we can pass a signed
		// integer to go backwards, it just goes "the long way round".
        uint64_t delta = (uint64_t) delta_;

        while (delta > 0) 
        {
        	// What's the least sig. bit of delta?
            if (delta & 1) 
            {
                acc_mult *= cur_mult;
                acc_plus = acc_plus * cur_mult + cur_plus;
            }
            cur_plus = (cur_mult + 1) * cur_plus;
            cur_mult *= cur_mult;
            delta /= 2;
        }

        state = acc_mult * state + acc_plus;
    }


	// Draw uniformly distributed permutation and permute the
	// given STL container
	// From: Knuth, TAoCP Vol. 2 (3rd 3d), Section 3.4.2     
    template <typename Iterator> 
    void shuffle(Iterator begin, Iterator end) 
    {
        for (Iterator it = end - 1; it > begin; --it)
            std::iter_swap(it, begin + get_bounded_rand((uint32_t) (it - begin + 1)));
    }

    /// Compute the distance between two PCG32 pseudorandom number generators
    int64_t operator-(const pcg32 &other) const 
    {
        assert(inc == other.inc);

        uint64_t
            cur_mult = PCG32_MULT,
            cur_plus = inc,
            cur_state = other.state,
            the_bit = 1u,
            distance = 0u;

        while (state != cur_state) 
        {
            if ((state & the_bit) != (cur_state & the_bit)) 
            {
                cur_state = cur_state * cur_mult + cur_plus;
                distance |= the_bit;
            }

            assert((state & the_bit) == (cur_state & the_bit));
            the_bit <<= 1;
            cur_plus = (cur_mult + 1ULL) * cur_plus;
            cur_mult *= cur_mult;
        }

        return (int64_t) distance;
    }

   	// Equality operator
    bool operator==(const pcg32 &other) const { return state == other.state && inc == other.inc; }

    // Inequality operator
    bool operator!=(const pcg32 &other) const { return state != other.state || inc != other.inc; }

    // RNG state.  All values are possible.
    uint64_t state;
    // Controls which RNG sequence (stream) is selected. Must *always* be odd.
    uint64_t inc;

    // Multiplication constant
    const unsigned long long PCG32_MULT = 0x5851f42d4c957f2d;

};

#endif // __PCG32_H
