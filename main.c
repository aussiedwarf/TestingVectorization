#include <stdio.h>
#include <time.h>

#include "xmmintrin.h"
#include "emmintrin.h"
#include "pmmintrin.h"

#if _WIN32 && defined(__GNUC__)
#include <sys/time.h>
#include <inttypes.h>
#endif

#if _MSC_VER
#define __restrict__ __restrict 

#define Align16 __declspec (align(16))
#else

#define Align16 __attribute__ ((aligned (16)))

#endif

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

int long long GetTime()
{
#if _WIN32 && defined(__GNUC__)	
  struct timeval t1;

	// start timer
	gettimeofday(&t1, NULL);

  return (int long long)t1.tv_sec * 1000000000LL + (int long long)t1.tv_usec * 1000LL;
#elif _MSC_VER
  struct _timespec32 t1;
  _timespec32_get(&t1, TIME_UTC);

  return (long long int)t1.tv_sec * 1000000000LL + (long long int)t1.tv_nsec;
#elif __APPLE__
  struct timespec t1;
  t1 = orwl_gettime();

  return (long long int)t1.tv_sec * 1000000000LL + (long long int)t1.tv_nsec;
#elif __linux__
  struct timespec t1;
  clock_gettime(CLOCK_MONOTONIC, &t1);

  return (long long int)t1.tv_sec * 1000000000LL + (long long int)t1.tv_nsec;
#else
  struct timespec t1;
  timespec_get(&t1, TIME_UTC);

  return (long long int)t1.tv_sec * 1000000000LL + (long long int)t1.tv_nsec;
#endif
}


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
  
  Align16 float multiplierA[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  Align16 float multiplierB[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  Align16 float result[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};

  int long long startTime[2];
  int long long stopTime[2];

  int i;
  
  printf("Begin\n");
  
  PrintMatricies(multiplierA,multiplierB,result);
  
  startTime[0] = GetTime();
  
#if _WIN32
  printf("Start Time: %I64d\n", startTime[0]);
#else
  printf("Start Time: %lld\n", startTime[0]);
#endif    
  float* ptrA = multiplierB;
  float* ptrB = result;
  
  for(i=0; i < (1<<iterations); i++)
  {
    MatrixMulFloat(multiplierA, ptrA, ptrB);
    float *t = ptrA;
    ptrA = ptrB;
    ptrB = t;
  }
  stopTime[0] = GetTime();

#if _WIN32
  printf("Stop Time:  %I64d\n", stopTime[0]);
#else
  printf("Stop Time:  %lld\n", stopTime[0]);
#endif    
  
  printf("\n");
  
  startTime[1] = GetTime();
#if _WIN32
  printf("Start Time: %I64d\n", startTime[1]);
#else
  printf("Start Time: %lld\n", startTime[1]);
#endif  
    
  ptrA = multiplierB;
  ptrB = result;
  
  for(i=0; i < (1<<iterations); i++)
  {
    MatrixMulFloatSSE(multiplierA, ptrA, ptrB);
    float *t = ptrA;
    ptrA = ptrB;
    ptrB = t;
  }
  stopTime[1] = GetTime();
#if _WIN32
  printf("Stop Time:  %I64d\n", stopTime[0]);
#else
  printf("Stop Time:  %lld\n", stopTime[0]);
#endif  
  
  PrintMatricies(multiplierA,multiplierB,result);
  
  printf("\n");
  printf("Result\n");
  
  long long int time = stopTime[0] - startTime[0];
#if _WIN32
  printf("Time:  %I64d\n", time);
#else
  printf("Time:  %lld\n", time);
#endif  

  
  time = stopTime[1] - startTime[1];
#if _WIN32
  printf("Time:  %I64d\n", time);
#else
  printf("Time:  %lld\n", time);
#endif  


  
  
  return 0;
  
}
