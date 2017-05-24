
# -march=native -S -o test -flto
# -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mtune=native
all:
	gcc main.c -Wall -lm -m64 -std=c11 -Ofast -ffast-math -funroll-loops -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mtune=native -o test

