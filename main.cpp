#include <iostream>

#include "include/pcg32.hpp"
#include "include/simd_pcg32.hpp"

int main()
{
	pcg32 gen;

	std::cout << "Some random numbers...\n";
	for(int i = 0; i < 10; ++i)
	{
		std::cout << gen.get_rand() << "\n";
	}

	pcg32_8 eight_gen;

	std::size_t N_rands = 50000;
	
	std::vector<uint32_t> rand_arr(N_rands);
	
	eight_gen.populate_array_avx2_pcg32(rand_arr.data(), N_rands);	

	// eightpopulate_array_avx2_pcg32

}