#include "include/benchmark.hpp"

int main()
{
	// Create a 32-bit PCG PRNG
	// The object is automatically seeded
	pcg32 gen;

	std::cout << "Some random numbers...\n";
	for(int i = 0; i < 10; ++i)
	{
		std::cout << gen.get_rand() << "\n";
	}

	// Used for benchmarking the different implementations
	benchmark my_bench;

	// Test each implementation
	my_bench.run_generators();
}