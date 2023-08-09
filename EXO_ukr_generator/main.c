#include <stdio.h>
#include <time.h>

#include "uk_exo.h"

#define M 8
#define N 12
#define K 512
static float A[K * M];
static float B[K * N];
static float C[N * M];
static float C2[N * M];
static float C3[N * M];
static float C4[N * M];

#define Aref(a1,a2)  A[ (a2)*(Alda)+(a1) ]
#define Bref(a1,a2)  B[ (a2)*(Blda)+(a1) ]
#define Cref(a1,a2)  C3[ (a2)*(Clda)+(a1) ]


void simplegemm(){
   int Alda = M, Clda =  M;
   int Blda = N;   
   int    i, j, p;
   for ( p=0; p<K; p++ )
	   for ( j=0; j<N; j++ )
		   for ( i=0; i<M; i++ )
			   Cref(i,j) = Cref(i,j) + Aref(i,p) * Bref(j,p);
}

void initialize() {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = (i * K + j);//*0.1;//3.2;
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i * N + j] = (i * N + j);//*0.2;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0.0;
      C2[i * N + j] = 0.0;
      C3[i * N + j] = 0.0;
      C4[i * N + j] = 0.0;
    }
  }
  return;
}

int main() {
  clock_t start, end;
  float msec;
  int reps=100000;
  initialize();
  double gflops = (2.0*M*N*K)/1e9;
  float alpha = 1.0;
  float beta = 1.0;

  printf("TEST STARTING...!\n");
  // Calling scheduled matmul
  start = clock();
  for (int i = 0; i < reps; i++){
      uk_8x12_a1True_b1True(NULL, K, &alpha, A,B, &beta, (struct exo_win_2f32){C2,{M,1}});
  }
    end = clock();

  msec = ((double)(end - start) / (double) CLOCKS_PER_SEC)/reps;
  for (int i = 0; i < reps; i++)
  simplegemm();
  for(int i = 0; i< M; i++)
  for(int j = 0; j< N; j++){
	  if(C2[j* M + i] == C3[j*M+i])
	  	 //printf("OK %f %f\n",C2[j*M+i],C3[j*M+i]);
		 continue;
	  else
	  	 printf("ERROR %f %f\n",C2[j*M+i],C3[j*M+i]);
  }
  printf("PASS!\n");
  return (0);
}
