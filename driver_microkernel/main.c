////////////////////////////////////////////////////////////////////////////////
/////     DRIVER FOR CALLING GEMM UK
/////
/////     Adrián Castelló (adcastel@disca.upv.es)
/////     October 2021
/////     Based on documentation at 
/////     github.com/flame/blis/blob/master/docs/KernelsHowTo.md#calling-kernels



#include <blis.h>

#include <arm_neon.h>
#include "gemm_blis_neon_fp32.h"
#include "uk_exo.h"
#include "blis_neon.h"
#include <sys/time.h>

#define Aref(a1,a2)  A[ (a2)*(Alda)+(a1) ]
#define Bref(a1,a2)  B[ (a2)*(Blda)+(a1) ]
#define Cref(a1,a2)  Cr[ (a2)*(Clda)+(a1) ]

#define  errorthd  1.0e-7

#if defined(FP32)
#define DTYPE float
#define BTYPE BLIS_FLOAT
#else
#define DTYPE double
#define BTYPE BLIS_DOUBLE
#endif

double dclock()
{
	/*
	 *  * Timer
	 *   *
	 *    */
	  struct timeval  tv;
	    // struct timezone tz;
	    
	       gettimeofday( &tv, NULL );
	    
	         return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}


void gemm_base( int m, int n, int k, DTYPE *A, int Alda, DTYPE *B, int Blda, DTYPE *Cr, int Clda ){
	/*
	 *    Baseline micro-kernel, for cases where the dimension does not match MR x NR
	 *    */
   int    i, j, p;
   for ( p=0; p<k; p++ )
          for ( j=0; j<n; j++ )
		for ( i=0; i<m; i++ )
		    Cref(i,j) = Cref(i,j) + Aref(i,p) * Bref(j,p);
}

int check_error(DTYPE * C, DTYPE * Cr, int size){
    int err=0;
    double error = 0.0;
    double nrm   = 0.0;
    double tmp;
    for(int i = 0; i < size; i++){
	 // printf("%f  -- %f\n",Cr[i], C[i]);
          tmp = (double) Cr[i]*Cr[i];
	  nrm += tmp*tmp;
	  tmp = (double) fabs(Cr[i]-C[i]);
	  error += tmp*tmp;
    }
    if ( nrm!=0.0 )
          error = sqrt(error) / sqrt(nrm);
    else
          error = sqrt(error);
    if ( error>errorthd )
         err = 1;
    return err;
}

int main(int argc, char * argv []){

	char* variant;
	char test;
	int M, N, PACKMR, PACKNR, PACKKR, PACKKC, PACKNC, PACKMC, K=2;
	
	bool row_pref;
	
	DTYPE  *A, *A2, *B; 
	DTYPE *C, *Cr;
        DTYPE alpha = 1.0, beta = 0.0;
	
	int kmin, kmax,kstep;
	int mmin, mmax,mstep;
	int nmin, nmax,nstep;
	int gemm_error;
	int nreps;
	char resident;
	int i;
	double t1, t2, time, tmin, flops, GFLOPS;
        
	FILE *fd_csv;

         	
	const cntx_t * cntx;
        auxinfo_t  aux;
#if defined(FP32)
        gemm_ukr_ft gemm_kernel;
#else	
        dgemm_ukr_ft gemm_kernel;
#endif
	variant= argv[1];
        int MR = atoi(argv[2]);
        int NR = atoi(argv[3]);	
	int KR=1;
	kmin  = atoi(argv[4]);
	kmax  = atoi(argv[5]);
	kstep = atoi(argv[6]);
	tmin   = atof(argv[7]);
	
        char name_file [30];
        sprintf(name_file,"output/%s_%d_%d.dat",argv[1],MR,NR);	
	fd_csv = fopen(name_file, "w");
         fprintf(fd_csv, "#%s %d %d %s\n",argv[1],MR,NR, "GFLOPS");
 printf("# ============================================================================================");
   printf("\n");
     printf("# Driver for the evaluation of uGEMM\n");
       printf("# ============================================================================================");
         printf("\n");
	   printf("#      library resident m     n     k     mr     nr     kr     mc     nc     kc     Time   GFLOPS");
	     printf("\n");
	         printf("# --------------------------------------------------------------------------------------------");
		   printf("\n");

        if (strcmp(variant,"NEON") == 0){
           /*mmin  = atoi(argv[7]);
	   mmax  = atoi(argv[8]);
	   mstep = atoi(argv[9]);

	   nmin  = atoi(argv[10]);
	   nmax  = atoi(argv[11]);
	   nstep = atoi(argv[12]);
	   
	   resident = argv[13][0];*/
	   mmin = mmax = MR;
	   nmin = nmax=  NR;
	   mstep = nstep = 1;
	   resident = 'C';
	}
        if (strcmp(variant,"EXO") == 0)
	{
	   mmin = mmax = MR;
	   nmin = nmax=  NR;
	   mstep = nstep = 1;
	   resident = 'C';

	}
        if (strcmp(variant,"BLIS") == 0)
	{
	
	    bli_init();
	    cntx = bli_gks_query_cntx();
            //We get the pointer to the GEMM using FLOATS
	     auxinfo_t aux; 
	    gemm_kernel = bli_cntx_get_l3_vir_ukr_dt(BTYPE, BLIS_GEMM_UKR, cntx);

            M = MR;//bli_cntx_get_blksz_def_dt(BTYPE, BLIS_MR, cntx);
	    N = NR; //bli_cntx_get_blksz_def_dt(BTYPE, BLIS_NR, cntx);
            mmin=mmax=M; mstep=1;
            nmin=nmax=N; nstep=1;
	    resident = 'C';
	    PACKMR = bli_cntx_get_blksz_max_dt(BTYPE, BLIS_MR, cntx);
	    PACKNR = bli_cntx_get_blksz_max_dt(BTYPE, BLIS_NR, cntx);
	    PACKKR = bli_cntx_get_blksz_max_dt(BTYPE, BLIS_KR, cntx);
	    PACKKC = bli_cntx_get_blksz_max_dt(BTYPE, BLIS_KC, cntx);
	    PACKMC = bli_cntx_get_blksz_max_dt(BTYPE, BLIS_MC, cntx);
	    PACKNC = bli_cntx_get_blksz_max_dt(BTYPE, BLIS_NC, cntx);

	    //row_pref = bli_cntx_get_l3_nat_ukr_prefs_dt(BTYPE, BLIS_GEMM_UKR, cntx);
	}
	A = (DTYPE *) malloc(sizeof(DTYPE)*mmax*kmax);
	A2 = (DTYPE *) malloc(sizeof(DTYPE)*mmax*kmax);
	B = (DTYPE *) malloc(sizeof(DTYPE)*kmax*nmax);
	C = (DTYPE *) malloc(sizeof(DTYPE)*mmax*nmax);
	
	for(i=0;i<mmax*kmax;i++) A2[i] = A[i] = ((DTYPE) rand())/RAND_MAX;
	for(i=0;i<kmax*nmax;i++) B[i] = ((DTYPE) rand())/RAND_MAX;
	for(i=0;i<mmax*nmax;i++){ C[i] = 0.0f;}
        
	for(int m=mmin;m<=mmax;m+=mstep){
	for(int n=nmin;n<=nmax;n+=nstep){
	for(int k=kmin;k<=kmax;k+=kstep){
        if (strcmp(variant,"BLIS") == 0)
	{
	  
	    K=k; //(k<=PACKKC)? k : PACKKC;
	    M=m;
	    N=n;
	    time  = 0.0;
	    t1    = dclock();
	    nreps = 0;
	    double taux;
	    int one=0;
	    while ( time <= tmin ) {
	       nreps+=100;
	       for(int i=0;i<100;i++){
	       gemm_kernel(M,N,K, &alpha, A, B, &beta, C, 1, M, &aux, cntx);
	       }
	       t2   = dclock();
	       time = ( t2 > t1 ? t2 - t1 : 0.0 );
	    }
	    time = time/nreps;
           flops   = 2.0 * m * n * k;
	   GFLOPS  = flops / (1.0e+9 * time );
           printf("        %5s  %c  %5d %5d %5d %6d %6d %6d %5d %5d %5d %8.2e %8.2e",
		             variant, resident, m, n, k, PACKMR, PACKNR, PACKKR, PACKMC, PACKNC, /*PACKKC*/K, time, GFLOPS);
	     printf("\n");


	} //BLIS
	else if(strcmp(variant,"NEON") == 0){
	    	time  = 0.0;
	    t1    = dclock();
	    nreps = 0;
	    double taux;
	    int one=0;
	    M = m; N = n; K = k;
	    while ( time <= tmin ) {
	       nreps+=100;
	      int ldC = M;
	      M = m; N = n; K = k;
	      
	      for(int i=0;i<100;i++)
	      gemm_microkernel_Cresident_neon_8x12_fp32( 'C', M, N, K, alpha, A, B, beta, C, ldC );
	      t2   = dclock();
	      one=!one;
	       time = ( t2 > t1 ? t2 - t1 : 0.0 );
	    }
	    time = time/nreps;
            flops   = 2.0 * m * n * k;
	    GFLOPS  = flops / (1.0e+9 * time );
           printf("        %5s  %c  %5d %5d %5d %6d %6d %6d %5d %5d %5d %8.2e %8.2e",
		             variant, resident, m, n, k, MR, NR, KR, m, n, k, time, GFLOPS);
	     printf("\n");
	}
	else if(strcmp(variant,"EXO") == 0){
	    time  = 0.0;
	    t1    = dclock();
	    nreps = 0;
	    double taux;
	    int one=0;
	    while ( time <= tmin ) {
	       nreps+=100;
	      int ldC = M;
	      M = m; N = n; K = k;
	      
	      for(int i=0;i<100;i++)
#if MMR == 8 
    #if NNR == 12
	      uk_8x12_a1True_b1True( NULL, K, &alpha, A, B, &beta, (struct exo_win_2f32){C,{M,1}});
    #elif NNR == 8
	      uk_8x8_a1True_b1True( NULL, K, &alpha, A, B, &beta, (struct exo_win_2f32){C,{M,1}});
    #elif NNR == 4
	      uk_8x4_a1True_b1True( NULL, K, &alpha, A, B, &beta, (struct exo_win_2f32){C,{M,1}});
   #endif	  
#elif MMR == 4 
    #if NNR == 12
	      uk_4x12_a1True_b1True( NULL, K, &alpha, A, B, &beta, (struct exo_win_2f32){C,{M,1}});
    #elif NNR == 8
	      uk_4x8_a1True_b1True( NULL, K, &alpha, A, B, &beta, (struct exo_win_2f32){C,{M,1}});
   #elif NNR == 4
	      uk_4x4_a1True_b1True( NULL, K, &alpha, A, B, &beta, (struct exo_win_2f32){C,{M,1}});
   #endif	      
#endif
	       t2   = dclock();
	      one=!one;
	       time = ( t2 > t1 ? t2 - t1 : 0.0 );
	    }
	    time = time/nreps;
            flops   = 2.0 * m * n * k;
	    GFLOPS  = flops / (1.0e+9 * time );
           printf("        %5s  %c  %5d %5d %5d %6d %6d %6d %5d %5d %5d %8.2e %8.2e",
		             variant, resident, m, n, k, MR, NR, KR, m, n, k, time, GFLOPS);
	     printf("\n");
	}
	
	
	} //k
	} //n
	} //m
	 fprintf(fd_csv, "%f\n", GFLOPS);
	 fclose(fd_csv);
       printf("# End of program...\n");
         printf("# ============================================================================================");
	   printf("\n");

        	
}
