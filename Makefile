CC = g++

CFLAGS= -O3 -march=native -Wall -Wextra -pedantic -Wshadow

SOURCES = main.cpp

EXECUTABLE = simd_pcg

benchmark: $(SOURCES)
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(SOURCES)

clean:
	rm -f $(EXECUTABLE)



