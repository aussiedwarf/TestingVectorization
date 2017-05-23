#include <stdio.h>
#include <time.h>

#include "xmmintrin.h"
#include "emmintrin.h"
#include "pmmintrin.h"

#if __APPLE__
#include <mach/mach_time.h>
#define ORWL_NANO (+1.0E-9)
#define ORWL_GIGA UINT64_C(1000000000)
static double orwl_timebase = 0.0;
static uint64_t orwl_timestart = 0;

struct timespec orwl_gettime(void) {
  // be more careful in a multithreaded environement
  if (!orwl_timestart) {
    mach_timebase_info_data_t tb = { 0 };
    mach_timebase_info(&tb);
    orwl_timebase = tb.numer;
    orwl_timebase /= tb.denom;
    orwl_timestart = mach_absolute_time();
  }
  struct timespec t;
  double diff = (mach_absolute_time() - orwl_timestart) * orwl_timebase;
  t.tv_sec = diff * ORWL_NANO;
  t.tv_nsec = diff - (t.tv_sec * ORWL_GIGA);
  return t;
}

#endif

#if 0
void MatrixTransposeFloat(const float * const __restrict__ a, float* __restrict__ result)
{
  result[ 0] = a[ 0];
  result[ 1] = a[ 4];
  result[ 2] = a[ 8];
  result[ 3] = a[12];
  
  result[ 4] = a[ 1];
  result[ 5] = a[ 5];
  result[ 6] = a[ 9];
  result[ 7] = a[13];
  
  result[ 8] = a[ 2];
  result[ 9] = a[ 6];
  result[10] = a[10];
  result[11] = a[14];
  
  result[12] = a[ 3];
  result[13] = a[ 7];
  result[14] = a[11];
  result[15] = a[15]; 
}

void MatrixMulTransFloat(const float * const __restrict__ a, const float * const __restrict__ b, float* __restrict__ result)
{
  float m[16] __attribute__ ((aligned (16)));
  MatrixTransposeFloat(a, m);
  
  result[ 0] = a[ 0]*b[ 0] + a[ 1]*b[ 1] + a[ 2]*b[ 2] + a[ 3]*b[ 3];
  result[ 1] = a[ 0]*b[ 4] + a[ 1]*b[ 5] + a[ 2]*b[ 6] + a[ 3]*b[ 7];
  result[ 2] = a[ 0]*b[ 8] + a[ 1]*b[ 9] + a[ 2]*b[10] + a[ 3]*b[11];
  result[ 3] = a[ 0]*b[12] + a[ 1]*b[13] + a[ 2]*b[14] + a[ 3]*b[15];
  
  result[ 4] = a[ 4]*b[ 0] + a[ 5]*b[ 1] + a[ 6]*b[ 2] + a[ 7]*b[ 3];
  result[ 5] = a[ 4]*b[ 4] + a[ 5]*b[ 5] + a[ 6]*b[ 6] + a[ 7]*b[ 7];
  result[ 6] = a[ 4]*b[ 8] + a[ 5]*b[ 9] + a[ 6]*b[10] + a[ 7]*b[11];
  result[ 7] = a[ 4]*b[12] + a[ 5]*b[13] + a[ 6]*b[14] + a[ 7]*b[15];
  
  result[ 8] = a[ 8]*b[ 0] + a[ 9]*b[ 1] + a[10]*b[ 2] + a[11]*b[ 3];
  result[ 9] = a[ 8]*b[ 4] + a[ 9]*b[ 5] + a[10]*b[ 6] + a[11]*b[ 7];
  result[10] = a[ 8]*b[ 8] + a[ 9]*b[ 9] + a[10]*b[10] + a[11]*b[11];
  result[11] = a[ 8]*b[12] + a[ 9]*b[13] + a[10]*b[14] + a[11]*b[15];
  
  result[12] = a[12]*b[ 0] + a[13]*b[ 1] + a[14]*b[ 2] + a[15]*b[ 3];
  result[13] = a[12]*b[ 4] + a[13]*b[ 5] + a[14]*b[ 6] + a[15]*b[ 7];
  result[14] = a[12]*b[ 8] + a[13]*b[ 9] + a[14]*b[10] + a[15]*b[11];
  result[15] = a[12]*b[12] + a[13]*b[13] + a[14]*b[14] + a[15]*b[15];
}
#endif

void MatrixMulFloat(const float * const __restrict__ a, const float * const __restrict__ b, float* __restrict__ result)
{
  result[ 0] = a[ 0]*b[ 0] + a[ 4]*b[ 1] + a[ 8]*b[ 2] + a[12]*b[ 3];
  result[ 1] = a[ 0]*b[ 4] + a[ 4]*b[ 5] + a[ 8]*b[ 6] + a[12]*b[ 7];
  result[ 2] = a[ 0]*b[ 8] + a[ 4]*b[ 9] + a[ 8]*b[10] + a[12]*b[11];
  result[ 3] = a[ 0]*b[12] + a[ 4]*b[13] + a[ 8]*b[14] + a[12]*b[15];
  
  result[ 4] = a[ 1]*b[ 0] + a[ 5]*b[ 1] + a[ 9]*b[ 2] + a[13]*b[ 3];
  result[ 5] = a[ 1]*b[ 4] + a[ 5]*b[ 5] + a[ 9]*b[ 6] + a[13]*b[ 7];
  result[ 6] = a[ 1]*b[ 8] + a[ 5]*b[ 9] + a[ 9]*b[10] + a[13]*b[11];
  result[ 7] = a[ 1]*b[12] + a[ 5]*b[13] + a[ 9]*b[14] + a[13]*b[15];
  
  result[ 8] = a[ 2]*b[ 0] + a[ 6]*b[ 1] + a[10]*b[ 2] + a[14]*b[ 3];
  result[ 9] = a[ 2]*b[ 4] + a[ 6]*b[ 5] + a[10]*b[ 6] + a[14]*b[ 7];
  result[10] = a[ 2]*b[ 8] + a[ 6]*b[ 9] + a[10]*b[10] + a[14]*b[11];
  result[11] = a[ 2]*b[12] + a[ 6]*b[13] + a[10]*b[14] + a[14]*b[15];
  
  result[12] = a[ 3]*b[ 0] + a[ 7]*b[ 1] + a[11]*b[ 2] + a[15]*b[ 3];
  result[13] = a[ 3]*b[ 4] + a[ 7]*b[ 5] + a[11]*b[ 6] + a[15]*b[ 7];
  result[14] = a[ 3]*b[ 8] + a[ 7]*b[ 9] + a[11]*b[10] + a[15]*b[11];
  result[15] = a[ 3]*b[12] + a[ 7]*b[13] + a[11]*b[14] + a[15]*b[15];
}

void MatrixMulFloatSSE(const float * const __restrict__ a, const float * const __restrict__ b, float* __restrict__ result)
{
  __m128 a_line, b_line, r_line, sum;
	
	a_line = _mm_load_ps(&a[0]);
	b_line = _mm_set1_ps(b[0]);
	sum = _mm_mul_ps(a_line,b_line);

	a_line = _mm_load_ps(&a[4]);
	b_line = _mm_set1_ps(b[1]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);

	a_line = _mm_load_ps(&a[8]);
	b_line = _mm_set1_ps(b[2]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	
	a_line = _mm_load_ps(&a[12]);
	b_line = _mm_set1_ps(b[3]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	_mm_store_ps(&result[0],sum);
	
	/*********************************************/

	a_line = _mm_load_ps(&a[0]);
	b_line = _mm_set1_ps(b[4]);
	sum = _mm_mul_ps(a_line,b_line);
	
	a_line = _mm_load_ps(&a[4]);
	b_line = _mm_set1_ps(b[5]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	
	a_line = _mm_load_ps(&a[8]);
	b_line = _mm_set1_ps(b[6]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	
	a_line = _mm_load_ps(&a[12]);
	b_line = _mm_set1_ps(b[7]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	_mm_store_ps(&result[4],sum);
	
	/*********************************************/
	
	a_line = _mm_load_ps(&a[0]);
	b_line = _mm_set1_ps(b[8]);
	sum = _mm_mul_ps(a_line,b_line);
	
	a_line = _mm_load_ps(&a[4]);
	b_line = _mm_set1_ps(b[9]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	
	a_line = _mm_load_ps(&a[8]);
	b_line = _mm_set1_ps(b[10]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	
	a_line = _mm_load_ps(&a[12]);
	b_line = _mm_set1_ps(b[11]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	_mm_store_ps(&result[8],sum);
	
	/*********************************************/
	
	a_line = _mm_load_ps(&a[0]);
	b_line = _mm_set1_ps(b[12]);
	sum = _mm_mul_ps(a_line,b_line);
	
	a_line = _mm_load_ps(&a[4]);
	b_line = _mm_set1_ps(b[13]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	
	a_line = _mm_load_ps(&a[8]);
	b_line = _mm_set1_ps(b[14]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	
	a_line = _mm_load_ps(&a[12]);
	b_line = _mm_set1_ps(b[15]);
	r_line = _mm_mul_ps(a_line,b_line);
	sum = _mm_add_ps(sum, r_line);
	_mm_store_ps(&result[12],sum);
}

void PrintMatricies(const float * const a, 
                    const float * const b, 
                    const float * const c)
{
  int i;
  for(i=0; i < 16; i++)
    printf("%f, ",a[i]);   
  printf("\n");
  
  for(i=0; i < 16; i++)
    printf("%f, ",b[i]);   
  printf("\n");
  
  for(i=0; i < 16; i++)
    printf("%f, ",c[i]);
  printf("\n");
}

//expected arguments, num iterations, file
int main(int argc, char *argv[])
{
  const int iterations = 30;
  
  float multiplierA[16] __attribute__ ((aligned (16))) = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  float multiplierB[16] __attribute__ ((aligned (16))) = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  float result[16] __attribute__ ((aligned (16))) = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  
  
  struct timespec startTime[2];
  struct timespec stopTime[2];
  
  int i;
  
  printf("Begin\n");
  
  PrintMatricies(multiplierA,multiplierB,result);
  
#if __APPLE__
  startTime[0] = orwl_gettime();
#elif __linux__
  clock_gettime(CLOCK_MONOTONIC, &startTime[0]);
#else
  timespec_get(&startTime[0], TIME_UTC);
#endif
  printf("Start Time: %ld, %ld\n", startTime[0].tv_sec, startTime[0].tv_nsec);
    
  float* ptrA = multiplierB;
  float* ptrB = result;
  
  for(i=0; i < (1<<iterations); i++)
  {
    MatrixMulFloat(multiplierA, ptrA, ptrB);
    float *t = ptrA;
    ptrA = ptrB;
    ptrB = t;
  }
#if __APPLE__  
  stopTime[0] = orwl_gettime();
#elif __linux__
  clock_gettime(CLOCK_MONOTONIC, &stopTime[0]);
#else
  timespec_get(&startTime[0], TIME_UTC);
#endif
  printf("Stop Time: %ld, %ld\n", stopTime[0].tv_sec, stopTime[0].tv_nsec);
  
  printf("\n");
  
#if __APPLE__  
  startTime[1] = orwl_gettime();
#elif __linux__
  clock_gettime(CLOCK_MONOTONIC, &startTime[1]);
#else
  timespec_get(&startTime[1], TIME_UTC);
#endif
  printf("Start Time: %ld, %ld\n", startTime[1].tv_sec, startTime[1].tv_nsec);
    
  ptrA = multiplierB;
  ptrB = result;
  
  for(i=0; i < (1<<iterations); i++)
  {
    MatrixMulFloatSSE(multiplierA, ptrA, ptrB);
    float *t = ptrA;
    ptrA = ptrB;
    ptrB = t;
  }
#if __APPLE__  
  stopTime[1] = orwl_gettime();
#elif __linux__
  clock_gettime(CLOCK_MONOTONIC, &stopTime[1]);
#else
  timespec_get(&startTime[1], TIME_UTC);
#endif
  printf("Stop Time: %ld, %ld\n", stopTime[1].tv_sec, stopTime[1].tv_nsec);
  
  PrintMatricies(multiplierA,multiplierB,result);
  
  printf("\n");
  printf("Result\n");
  
  long long int start = (long long int)startTime[0].tv_sec * 1000000000LL + (long long int)startTime[0].tv_nsec;
  long long int stop = (long long int)stopTime[0].tv_sec * 1000000000LL + (long long int)stopTime[0].tv_nsec; 
  long long int time = stop - start;
  printf("Time: %lld\n", time);
  
  start = (long long int)startTime[1].tv_sec * 1000000000LL + (long long int)startTime[1].tv_nsec;
  stop = (long long int)stopTime[1].tv_sec * 1000000000LL + (long long int)stopTime[1].tv_nsec; 
  time = stop - start;
  printf("Time: %lld\n", time);
  
  
  
  

  
  
  return 0;
  
}
