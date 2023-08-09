/* 
   GEMM FLAVOURS

   -----

   GEMM FLAVOURS is a family of algorithms for matrix multiplication based
   on the BLIS approach for this operation: https://github.com/flame/blis

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   -----

   author    = "Enrique S. Quintana-Orti"
   contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include "dtypes.h"
#include "gemm_blis.h"
#include "inutils.h"
#include "colors.h"
#include "../modelLevel/model_level.h"

#include "blis.h"

#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )
#define min(a,b)     ( (a) > (b) ? (b) : (a) )
#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Cgcol(a1,a2) Cg[ (a2)*(ldC)+(a1) ]
#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Cgrow(a1,a2) Cg[ (a1)*(ldC)+(a2) ]

extern int    print_matrix( char *, char, size_t, size_t, DTYPE *, size_t );
extern int    generate_matrix( char, size_t, size_t, DTYPE *, size_t );
extern double dclock();
void gemm( char, char, char, char, char, size_t, size_t, size_t, DTYPE, DTYPE *, int, DTYPE *, int, DTYPE, DTYPE *, int );

void *my_aligned_alloc(size_t alignment, size_t size, int zero) {
    size_t request_size = size + alignment;
    char* buf = (char*)(zero ? calloc(1, request_size) : malloc(request_size));

    size_t remainder = ((size_t)buf) % alignment;
    size_t offset = alignment - remainder;
    char* ret = buf + (unsigned char)offset;

    // store how many extra bytes we allocated in the byte just before the
    // pointer we return
    *(unsigned char*)(ret - 1) = offset;

    return (void*)ret;
}

/* Free memory allocated with aligned_alloc */
void my_aligned_free(void* aligned_ptr) {
    int offset = *(((char*)aligned_ptr) - 1);
    free(((char*)aligned_ptr) - offset);
}

int main(int argc, char *argv[]) {
  #if !defined(BLIS)
    //  BLIS MICRO-KERNEL 
    // -------------------
    #define BTYPE BLIS_FLOAT
    auxinfo_t aux;
    bli_init();
    const cntx_t * cntx = bli_gks_query_cntx();
    gemm_ukr_ft gemm_kernel = bli_cntx_get_l3_vir_ukr_dt(BLIS_FLOAT, BLIS_GEMM_UKR, cntx);
  #endif
  
  char  orderA, orderB, orderC, transA, transB, test;
  char variant[20];
  DTYPE *A  = NULL, 
	*B  = NULL, 
	*C  = NULL, 
	*Cg = NULL, 
	*Ac = NULL, 
	*Bc = NULL, 
	*Cc = NULL;
  DTYPE alpha, beta;
  double t1, t2, time, tmin, error, nrm, tmp, errorthd, flops, GFLOPS;
  size_t i, j, nreps, 
         visual, ldA, ldB, ldC;
 
  size_t m, n, k, mc, nc, kc, mmin, mmax, mstep, nmin, nmax, nstep, kmin, kmax, kstep; 
  #if defined(FP16)
    errorthd = 1.0e-3;
  #elif defined(FP32)
    errorthd = 1.0e-6;
  #elif defined(FP64)
    errorthd = 1.0e-14;
  #endif


#if defined(FAMILY)
  sprintf(variant, "ALG+NEON");  
#elif defined(BLIS)
  sprintf(variant, "BLIS");  
#elif defined(FAMILY_BLIS)
  sprintf(variant, "ALG+BLIS");  
#elif defined(FAMILY_EXO)
  sprintf(variant, "ALG+EXO");  
#endif

  orderA   = argv[2][0];
  orderB   = argv[3][0];
  orderC   = argv[4][0];
  
  transA   = argv[5][0];
  transB   = argv[6][0];
  
  alpha    = atof(argv[7]);
  beta     = atof(argv[8]);
  
  mmin     = atoi(argv[9]);
  mmax     = atoi(argv[10]);
  mstep    = atoi(argv[11]);
  
  nmin     = atoi(argv[12]);
  nmax     = atoi(argv[13]);
  nstep    = atoi(argv[14]);
  
  kmin     = atoi(argv[15]);
  kmax     = atoi(argv[16]);
  kstep    = atoi(argv[17]);
  
  visual   = atoi(argv[18]);
  tmin     = atof(argv[19]);
  test     = argv[20][0];

  FILE *fd_csv;
  
  testConfig_t* testConf = NULL;
  unsigned int cnn_num = 1;
  unsigned int cnn_enable = 0;
  
  if ((mmax == 0) && (nmax == 0) && (kmax == 0)) {
    testConf = new_CNN_Test_Config(argv);
    cnn_num = testConf->cnn_num;
    cnn_enable = 1;
  }


  fd_csv = fopen(argv[argc - 1], "w");
  fprintf(fd_csv, "#l;m;n;k;Gflops\n");

  
  printf(" ===========================================================================================\n");
  printf(" |                                  %sRUN CONFIGURATION %s                                     \n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf(" ===========================================================================================\n");
  printf(" |  [*] Data Type    : ");
  #ifdef FP32
    printf(" %sFloat%s                                                              \n", COLOR_BOLDWHITE, COLOR_RESET);
  #else
    printf(" %sUnknown%s                                                                            \n", COLOR_BOLDWHITE, COLOR_RESET);
  #endif
  //----------------------------------------------------------------------------------------------------------------------
  printf(" |  [*] Mode Selected: ");
  #ifdef FAMILY
    printf(" %sALG+NEON%s                                                             \n", COLOR_BOLDWHITE, COLOR_RESET);
  #elif FAMILY_BLIS
    printf(" %sALG+BLIS %s                                                              \n", COLOR_BOLDWHITE, COLOR_RESET);
  #elif FAMILY_EXO
    printf(" %sALG+EXO %s                                                              \n", COLOR_BOLDWHITE, COLOR_RESET);
  #elif BLIS
    printf(" %sBLIS%s                                                                               \n", COLOR_BOLDWHITE, COLOR_RESET);
  #else
    printf(" %sUnknown%s                                                                            \n", COLOR_BOLDWHITE, COLOR_RESET);
  #endif
  //----------------------------------------------------------------------------------------------------------------------
  printf(" |  [*] SIMD Selected: ");
  #ifdef AVX2
    printf(" %sAVX2%s                                                                \n", COLOR_BOLDWHITE, COLOR_RESET);
  #elif ARMv8
    printf(" %sARMv8%s                                                              \n", COLOR_BOLDWHITE, COLOR_RESET);
  #elif ARMv8_EXO
    printf(" %sARMv8_EXO%s                                                           \n", COLOR_BOLDWHITE, COLOR_RESET);
  #else
    printf(" %sUnknown%s                                                             \n", COLOR_BOLDWHITE, COLOR_RESET);
  #endif
  //----------------------------------------------------------------------------------------------------------------------
  printf(" |  [*] Dataset      :  %s%-24s%s                                           \n", COLOR_BOLDWHITE, argv[21], COLOR_RESET);
  printf(" |  [*] Output       :  %s%-22s%s                                           \n", COLOR_BOLDWHITE, argv[22], COLOR_RESET);
  //----------------------------------------------------------------------------------------------------------------------
  #ifndef BLIS
    printf(" |  [*] MR           :  %s%-5d%s                                                              \n", COLOR_BOLDWHITE, MR, COLOR_RESET);
    printf(" |  [*] NR           :  %s%-5d%s                                                              \n", COLOR_BOLDWHITE, NR, COLOR_RESET);
  #endif
  //----------------------------------------------------------------------------------------------------------------------
  printf(" |  [*] Tmin         :  %s%-5.2f%s                                                              \n", COLOR_BOLDWHITE, tmin, COLOR_RESET);
  printf(" ------------------------------------------------------------------------------------------\n");


  printf("\n"); 
  printf(" ==========================================================================================\n");
  printf(" |                           %sDRIVER FOR THE EVALUATION OF GEMM%s                            |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf(" ==========================================================================================\n");
  printf(" |  %sOrder  Trans      M       N      K     MC      NC     KC%s   |    %sTime      GFLOPS%s      |\n", COLOR_BOLDWHITE, COLOR_RESET, COLOR_BOLDWHITE, COLOR_RESET );
  printf(" ------------------------------------------------------------------------------------------\n");


  for (unsigned int cnn_i = 0; cnn_i < cnn_num; cnn_i++) {
    if (cnn_enable) {
    
      mmin  = testConf->cnn[cnn_i].mmin;
      mmax  = testConf->cnn[cnn_i].mmax;
      mstep = testConf->cnn[cnn_i].mstep;
      
      nmin  = testConf->cnn[cnn_i].nmin;
      nmax  = testConf->cnn[cnn_i].nmax;
      nstep = testConf->cnn[cnn_i].nstep;
      
      kmin  = testConf->cnn[cnn_i].kmin;
      kmax  = testConf->cnn[cnn_i].kmax;
      kstep = testConf->cnn[cnn_i].kstep;
    }
    
    //------------------------------------------------------------------------
    // MODEL VALUES: MC, NC and KC
    //------------------------------------------------------------------------
    int mc_tmp, nc_tmp, kc_tmp;
    get_optim_mc_nc_kc(sizeof(DTYPE), mmax, nmax, kmax, MR, NR, &mc_tmp, &nc_tmp, &kc_tmp);
    
    mc = (size_t)mc_tmp;
    nc = (size_t)nc_tmp;
    kc = (size_t)kc_tmp;

    //MC, NC, KC: BLIS Optims
    //mc = 144;
    //nc = 4080;
    //kc = 256;
    //-------------------------------------------------------------------------
    
    /*    
    A = (DTYPE *) malloc( mmax*kmax*sizeof(DTYPE) );   
    B = (DTYPE *) malloc( kmax*nmax*sizeof(DTYPE) );   
    C = (DTYPE *) malloc( mmax*nmax*kmax*sizeof(DTYPE) );
    
    Ac = (DTYPE *) malloc( (MR+mc)*(KR+kc)*sizeof(DTYPE) );   
    Bc = (DTYPE *) malloc( (KR+kc)*(NR+nc)*sizeof(DTYPE) );   
    Cc = (DTYPE *) malloc( (MR+mc)*(NR+nc)*sizeof(DTYPE) );
    */
   
    A  = (DTYPE *)my_aligned_alloc(32, mmax*kmax*sizeof(DTYPE),      1);   
    B  = (DTYPE *)my_aligned_alloc(32, kmax*nmax*sizeof(DTYPE),      1);   
    C  = (DTYPE *)my_aligned_alloc(32, mmax*nmax*sizeof(DTYPE), 1);

    Ac = (DTYPE *)my_aligned_alloc(32, (MR+mc)*(KR+kc)*sizeof(DTYPE), 1);
    Bc = (DTYPE *)my_aligned_alloc(32, (KR+kc)*(NR+nc)*sizeof(DTYPE), 1);
    Cc = (DTYPE *)my_aligned_alloc(32, (MR+mc)*(NR+nc)*sizeof(DTYPE), 1);
    
    if ( test=='T' )
      Cg = (DTYPE *) malloc(mmax*nmax*sizeof(DTYPE));   
    

    for ( m=mmin; m<=mmax; m+=mstep ){
      for ( n=nmin; n<=nmax; n+=nstep ){
	for ( k=kmin; k<=kmax; k+=kstep ){
	  
	  if ( ((transA=='N')&&(orderA=='C'))||
	       ((transA=='T')&&(orderA=='R')) )
	    ldA = m;
	  else
	    ldA = k;
	  
	  if ( transA=='N' ){
	    generate_matrix( orderA, m, k, A, ldA );
	  }else{
	    generate_matrix( orderA, k, m, A, ldA );
	  }
	  
	  if ( ((transB=='N')&&(orderB=='C'))||
	       ((transB=='T')&&(orderB=='R')) )
	    ldB = k;
	  else
	    ldB = n;

	  if ( transB=='N' )
	    generate_matrix( orderB, k, n, B, ldB );
	  else
	    generate_matrix( orderB, n, k, B, ldB );
	  
	  if ( orderC=='C' )
	    ldC = m;
	  else
	    ldC = n;

	  generate_matrix( orderC, m, n, C, ldC );
	  
	  if ( test=='T' ) {
	    if ( orderC=='C' )
	      for ( j=0; j<n; j++ )
		for ( i=0; i<m; i++ ) 
		  Cgcol(i,j)=Ccol(i,j);
	    else
	      for ( j=0; j<n; j++ )
		for ( i=0; i<m; i++ ) 
		  Cgrow(i,j)=Crow(i,j);
	  }
	  
	  // Print data
	  if ( visual == 1 ) {
	    if ( transB=='N' ) 
	      print_matrix( "Ai", orderA, m, k, A, ldA );
	    else
	      print_matrix( "Ai", orderA, k, m, A, ldA );
	    if ( transB=='N' ) 
	      print_matrix( "Bi", orderB, k, n, B, ldB );
	    else
	      print_matrix( "Bi", orderB, n, k, B, ldB );
	    print_matrix( "Ci", orderC, m, n, C, ldC );
	  }
	  
	  
	  time  = 0.0; 
	  t1    = dclock();
	  nreps = 0;
	  while ( time <= tmin ) {
	    nreps++;
            #if defined(FAMILY) || defined(FAMILY_BLIS) || defined(FAMILY_EXO)
	      gemm_blis_B3A2C0( orderA, orderB, orderC, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, 
	      		        Ac, Bc, mc, nc, kc, cntx, &aux, gemm_kernel);
            #else
	      sgemm_("No transpose", "No transpose", (void *)&m, (void *)&n, (void *)&k, &alpha, A, (void *)&ldA, B, 
		    (void *)&ldB, &beta, C, (void *)&ldC);
            #endif
	    t2   = dclock();
	    time = ( t2 > t1 ? t2 - t1 : 0.0 );
	  }
	  time = time/nreps;
	  
	  // Test result
	  if ( test=='T' ) {
	    gemm( orderA, orderB, orderC, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, Cg, ldC );
	    error = 0.0;
	    nrm   = 0.0;
	    if ( orderC=='C' )
	      for ( j=0; j<n; j++ )
		for ( i=0; i<m; i++ ) {
		  tmp = (double) Cgcol(i,j)*Cgcol(i,j);
		  nrm += tmp*tmp;
		  tmp = (double) dabs(Cgcol(i,j)-Ccol(i,j)); 
		  error += tmp*tmp;
		}
	    else
	      for ( j=0; j<n; j++ )
		for ( i=0; i<m; i++ ) {
		  tmp = (double) Cgrow(i,j)*Cgrow(i,j);
		  nrm += tmp*tmp;
		  tmp = (double) dabs(Cgrow(i,j)-Crow(i,j)); 
		  error += tmp*tmp;
		}
	    if ( nrm!=0.0 )
	      error = sqrt(error) / sqrt(nrm);
	    else
	      error = sqrt(error);
	  }
	  else
	    error = -1.0;
	  
	  // Print results
	  if ( visual == 1 ) {
	    print_matrix( "Cf", orderC, m, n, C, ldC );
	    if (test == 'T')
              print_matrix( "Crf", orderC, m, n, Cg, ldC );
	  }	  
	  //printf("-->Results\n");
	  //printf("   Time         = %12.6e seg.\n", time  );
	  flops   = 2.0 * m * n * k;
	  GFLOPS  = flops / (1.0e+9 * time );

          #ifdef BLIS
	  mc = 0; nc = 0; kc = 0;
          #endif
	  printf(" |   %c%c%c    %c%c     %6zu %6zu %6zu %6zu %6zu %6zu   |  %8.2e %10.2e     |", 
		 orderA, orderB, orderC, transA, transB, m, n, k, mc, nc, kc, time, GFLOPS );
	  
	  printf("\n");
	  if (cnn_enable)
	    fprintf(fd_csv, "%d;%zu;%zu;%zu;%.2e\n", testConf->cnn[cnn_i].layer, m, n, k, GFLOPS);
	  //else
	  //fprintf(fd_csv, "%d;%d;%d;%d;%.2e\n", 0, m, n, k, GFLOPS);
	}
      }
    }

    my_aligned_free(A);
    my_aligned_free(B);
    my_aligned_free(C);
    my_aligned_free(Ac);
    my_aligned_free(Bc);
    my_aligned_free(Cc);
  
    if ( test=='T' )
      free(Cg);
  }
  
  printf(" ==========================================================================================\n");
  printf("\n");
  return 0;
}

void gemm( char orderA, char orderB, char orderC, 
           char transA, char transB, 
           size_t m, size_t n, size_t k, 
           DTYPE alpha, DTYPE *A, int ldA, 
	                DTYPE *B, int ldB, 
           DTYPE beta,  DTYPE *C, int ldC ){

   size_t    ic, jc, i, j, p;
   DTYPE  zero = 0.0, one = 1.0, tmp;

   // Quick return if possible
  if ( (m==0)||(n==0) || (((alpha==zero) || (k==0)) && (beta==one)) )
    return;

  #include "quick_gemm.h"

  if ( (transA=='N')&&(transB=='N') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Bcol(p,j);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Brow(p,j);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Bcol(p,j);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Brow(p,j);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='N')&&(transB=='T') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Bcol(j,p);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Brow(j,p);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Bcol(j,p);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Brow(j,p);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='T')&&(transB=='N') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Bcol(p,j);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Brow(p,j);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Bcol(p,j);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Brow(p,j);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='T')&&(transB=='T') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Bcol(j,p);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Brow(j,p);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Bcol(j,p);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Brow(j,p);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else {
    printf("Error: Invalid options for transA, transB: %c %c\n", transA, transB);
    exit(-1);
  }
}


