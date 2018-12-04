#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstring>
#include <iostream>
#include <array>
#include <random>
#include <iomanip>
#include <immintrin.h>

#include "randutils.hpp"

#include "pcg32.hpp"

#if defined(__AVX2__)
#include "simd_avx2_pcg32.hpp"
#include "simd_avx256_pcg32.hpp"
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include "simd_avx512_pcg32.hpp"
#endif

class benchmark
{
protected:
	pcg32 my_pcg32;

#if defined(__AVX2__)
	simd_avx2_pcg32 my_avx2_pcg32;
	simd_avx256_pcg32 my_avx256_pcg32;
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
	simd_avx512_pcg32 my_avx512_pcg32;
#endif

	// Number of random numbers to generate
	const std::size_t N_rands = 50000;
	
	// The number of  random numbers used in the shuffling
	// benchmakr
	const std::size_t N_shuffle = 10000;
	
	// The number of times to repeat each function benchmark
	const std::size_t repeats = 500;
	
	// Use a vector here in case a lot of rands are requested
	std::vector<uint32_t> rand_arr;

	// Benchmarking functions
    void RDTSC_start(uint64_t* cycles)
    {
        // (Hopefully) place these in a register
        register unsigned cyc_high, cyc_low;

        // Use the read time-stamp counter so we can count the 
        // number of cycles
        __asm volatile("cpuid\n\t"                                               
                       "rdtsc\n\t"                                               
                       "mov %%edx, %0\n\t"                                       
                       "mov %%eax, %1\n\t"                                       
                       : "=r"(cyc_high), "=r"(cyc_low)::"%rax", "%rbx", "%rcx",  
                         "%rdx");    

        // Update the number of cycles
        *cycles = (uint64_t(cyc_high) << 32) | cyc_low; 
    }

    void RDTSC_final(uint64_t* cycles)
    {
        // (Hopefully) place these in a register
        register unsigned cyc_high, cyc_low;

        // Use the read time-stamp counter so we can count the 
        // number of cycles
        __asm volatile("rdtscp\n\t"                                               
                       "mov %%edx, %0\n\t"
                       "mov %%eax, %1\n\t"
                       "cpuid\n\t"
                       : "=r"(cyc_high), "=r"(cyc_low)::"%rax", "%rbx", "%rcx",
                         "%rdx");

        *cycles = (uint64_t(cyc_high) << 32) | cyc_low;                          
    }


    // This function can be passed a member function and an array for testing 
    template <typename TEST_CLASS>
    void benchmark_fn(void (TEST_CLASS::*test_fn)(uint32_t*, uint32_t), TEST_CLASS& class_obj, uint32_t* test_array, const std::string& str,
    																						const std::size_t size)
    {
    	std::fflush(nullptr);

        uint64_t cycles_start{0}, cycles_final{0}, cycles_diff{0};

        // Get max value of a uin64_t by rolling it over
        uint64_t min_diff = (uint64_t)-1;

        std::cout << "Testing function : " << str << "\n"; 

        for(uint32_t i = 0; i < repeats; i++)
        {
            // Pretend to clobber memory 
            // Don't allow reordering of memory access instructions
            __asm volatile("" ::: "memory");
            
            RDTSC_start(&cycles_start);
         
            (class_obj.*test_fn)(test_array, size);
            
            RDTSC_final(&cycles_final);

            cycles_diff = (cycles_final - cycles_start);   
            
            if (cycles_diff < min_diff)
                min_diff = cycles_diff;
        }  

        uint64_t S = N_rands;

        // Calculate the number of cycles per operation
        float cycles_per_op = min_diff / float(S);

        std::cout << std::setprecision(2) << cycles_per_op << " cycles per operation\n";
        std::fflush(nullptr);
    }


public:
	
	benchmark() {rand_arr.resize(N_rands);}     

	void run_generators()
    {
		std::cout << "\n==========================\n" <<
					   		"\tGenerators" 			  <<
					 "\n==========================\n\n";

    	std::string fn_name = "populate_array_pcg32";
    	benchmark_fn(&pcg32::populate_array_pcg32, my_pcg32, rand_arr.data(), fn_name, N_rands);

   	#if defined(__AVX2__)
    	fn_name = "populate_array_simd_pcg32";
    	benchmark_fn(&simd_avx2_pcg32::populate_array_simd_pcg32, my_avx2_pcg32, rand_arr.data(), fn_name, N_rands);

    	fn_name = "populate_array_simd_pcg32_two";
    	benchmark_fn(&simd_avx2_pcg32::populate_array_simd_pcg32_two, my_avx2_pcg32, rand_arr.data(), fn_name, N_rands);

    	fn_name = "populate_array_simd_pcg32_four";
    	benchmark_fn(&simd_avx2_pcg32::populate_array_simd_pcg32_four, my_avx2_pcg32, rand_arr.data(), fn_name, N_rands);

    	fn_name = "populate_array_avx256_pcg32";
    	benchmark_fn(&simd_avx256_pcg32::populate_array_avx256_pcg32, my_avx256_pcg32, rand_arr.data(), fn_name, N_rands);

    	fn_name = "populate_array_avx256_pcg32_two";
    	benchmark_fn(&simd_avx256_pcg32::populate_array_avx256_pcg32_two, my_avx256_pcg32, rand_arr.data(), fn_name, N_rands);

    	fn_name = "populate_array_avx256_pcg32_two";
    	benchmark_fn(&simd_avx256_pcg32::populate_array_avx256_pcg32_two, my_avx256_pcg32, rand_arr.data(), fn_name, N_rands);
	#endif

	#if defined(__AVX512F__) && defined(__AVX512DQ__)
    	fn_name = "populate_array_avx512_pcg32";
    	benchmark_fn(&simd_avx512_pcg32::populate_array_avx512_pcg32, my_avx512_pcg32, rand_arr.data(), fn_name, N_rands);

    	fn_name = "populate_array_avx512_pcg32_two";
    	benchmark_fn(&simd_avx512_pcg32::populate_array_avx512_pcg32_two, my_avx512_pcg32, rand_arr.data(), fn_name, N_rands);
		
		fn_name = "populate_array_avx512_pcg32_four";
    	benchmark_fn(&simd_avx512_pcg32::populate_array_avx512_pcg32_four, my_avx512_pcg32, rand_arr.data(), fn_name, N_rands);

    	fn_name = "populate_array_avx512bis_pcg32";
    	benchmark_fn(&simd_avx512_pcg32::populate_array_avx512bis_pcg32, my_avx512_pcg32, rand_arr.data(), fn_name, N_rands);

    	fn_name = "populate_array_avx512bis_pcg32_two";
    	benchmark_fn(&simd_avx512_pcg32::populate_array_avx512bis_pcg32_two, my_avx512_pcg32, rand_arr.data(), fn_name, N_rands);

    	fn_name = "populate_array_avx512bis_pcg32_four";
    	benchmark_fn(&simd_avx512_pcg32::populate_array_avx512bis_pcg32_four, my_avx512_pcg32, rand_arr.data(), fn_name, N_rands);
    #endif


    }
    
};

#endif