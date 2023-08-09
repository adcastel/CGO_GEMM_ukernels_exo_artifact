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

#include "gemm_blis.h"

int print_matrix(char *, char, size_t, int, DTYPE *, size_t);


void gemm_blis_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB, 
                       size_t m, size_t n, size_t k, 
                       DTYPE alpha, DTYPE *A, size_t ldA, 
		                    DTYPE *B, size_t ldB, 
		       DTYPE beta,  DTYPE *C, size_t ldC, 
		       DTYPE *Ac, DTYPE *Bc, 
                       size_t MC, size_t NC, size_t KC,
		       const cntx_t * cntx, auxinfo_t * aux, gemm_ukr_ft gemm_kernel) {
  size_t    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  DTYPE  zero = 0.0, one = 1.0, betaI; 
  DTYPE  *Aptr, *Bptr, *Cptr;

  #if defined(CHECK)
  #include "check_params.h"
  #endif
   
  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;
  
  #include "quick_gemm.h"
  
  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 
      
      if ( (transB=='N')&&(orderB=='C') )
        Bptr = &Bcol(pc,jc);
      else if ( (transB=='N')&&(orderB=='R') )
        Bptr = &Brow(pc,jc);
      else if ( (transB=='T')&&(orderB=='C') )
        Bptr = &Bcol(jc,pc);
      else
        Bptr = &Brow(jc,pc);
      
      pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR);
      
      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;
      
      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 
	
        if ( (transA=='N')&&(orderA=='C') ){
          Aptr = &Acol(ic, pc);
	}else if ( (transA=='N')&&(orderA=='R') ){
          Aptr = &Arow(ic, pc);
	}else if ( (transA=='T')&&(orderA=='C') ){
          Aptr = &Acol(pc, ic);
	}else{
          Aptr = &Arow(pc, ic);
	}
	
	//Comment or uncomment for packing or not
        pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);
	
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 
	  
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 
	    
            if ( orderC=='C' )
              Cptr = &Ccol(ic+ir,jc+jr);
	    else
              Cptr = &Crow(ic+ir,jc+jr);

            #if defined(FAMILY_EXO)
	    if (mr == 8 && nr == 12)
                uk_8x12_a1True_b1True( NULL, kc, &alpha,  &Ac[ir*kc],  &Bc[jr*kc], &betaI, (struct exo_win_2f32){Cptr,{m,1}});
	    else if (mr == 8 && nr == 4)
                uk_8x4_a1True_b1True( NULL, kc, &alpha,  &Ac[ir*kc],  &Bc[jr*kc], &betaI, (struct exo_win_2f32){Cptr,{m,1}});
	    else if(mr==8 && nr == 8)
                uk_8x8_a1True_b1True( NULL, kc, &alpha,  &Ac[ir*kc],  &Bc[jr*kc], &betaI, (struct exo_win_2f32){Cptr,{m,1}});
	    else if(mr==4 && nr == 12)
                uk_4x12_a1True_b1True( NULL, kc, &alpha,  &Ac[ir*kc],  &Bc[jr*kc], &betaI, (struct exo_win_2f32){Cptr,{m,1}});
	    else if (mr == 4 && nr == 8)
                uk_4x8_a1True_b1True( NULL, kc, &alpha,  &Ac[ir*kc],  &Bc[jr*kc], &betaI, (struct exo_win_2f32){Cptr,{m,1}});
	    else if (mr == 4 && nr == 4)
                uk_4x4_a1True_b1True( NULL, kc, &alpha,  &Ac[ir*kc],  &Bc[jr*kc], &betaI, (struct exo_win_2f32){Cptr,{m,1}});
	    else
                uk_1xX_a1True_b1True( NULL, kc, &alpha,  &Ac[ir*kc],  &Bc[jr*kc], &betaI, Cptr);
            #else
                #if defined(FAMILY_BLIS)
	         gemm_kernel(mr, nr, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI,  Cptr, 1, ldC, aux, cntx);
                #else
                  #ifdef AVX2
                    #if (MR == 16) && (NR == 6)
	              gemm_microkernel_Cresident_AMD_avx256_2vx6_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC ); 
                    #else
		      printf("ERROR: Micro-kernel not implemented\n");
		     exit(-1);
                    #endif
	           #elif ARMv8
                     #if (MR == 8) && (NR == 12)
	                gemm_microkernel_Cresident_neon_8x12_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC ); 
                    #else
		     printf("ERROR: Micro-kernel not implemented\n");
		  exit(-1);
                #endif
              #endif
              #endif
	    #endif

          }
	  
        }
      }
    }
  }
}


void pack_RB( char orderM, char transM, int mc, int nc, DTYPE *M, int ldM, DTYPE *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, ii, k, rr;

  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) ) {
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
	  Mc[k] = Mcol(i+ii, j);
          k++;
        } 
        k += (RR-rr);
      }
    }
  } else {
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
           Mc[k] = Mcol(j,i+ii);
          k++;
        }
        k += (RR-rr);
      }
    }
  }
}

void pack_CB( char orderM, char transM, int mc, int nc, DTYPE *M, int ldM, DTYPE *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, jj, k, nr;

  k = 0;
  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mc[k] = Mcol(i,j+jj);
          k++;
        }
        k += (RR-nr);
      }
    }
  else
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mc[k] = Mcol(j+jj,i);
          k++;
        }
        k += (RR-nr);
      }
    }
}

void unpack_RB( char orderM, char transM, int mc, int nc, DTYPE *M, int ldM, DTYPE *Mc, int RR ){
/*
  BLIS unpack for M-->Mc
*/
  int    i, j, ii, k, rr;

  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) ) {
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          Mcol(i+ii,j) = Mc[k];
          k++;
        }
        k += (RR-rr);
      }
    }
  } else {
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          Mcol(j,i+ii) = Mc[k];
          k++;
        }
        k += (RR-rr);
      }
    }
  }
}

void unpack_CB( char orderM, char transM, int mc, int nc, DTYPE *M, int ldM, DTYPE *Mc, int RR ){
/*
  BLIS unpack for M-->Mc
*/
  int    i, j, jj, k, nr;

  k = 0;
  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mcol(i,j+jj) = Mc[k];
          k++;
        }
        k += (RR-nr);
      }
    }
  else
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mcol(j+jj,i) = Mc[k];
          k++;
        }
        k += (RR-nr);
      }
    }
}

void gemm_base_Cresident( char orderC, int m, int n, int k, 
                          DTYPE alpha, DTYPE *A, int ldA, 
                                       DTYPE *B, int ldB, 
                          DTYPE beta,  DTYPE *C, int ldC ){
/*
  Baseline micro-kernel 
  Replace with specialized micro-kernel where C-->m x n is resident in registers
*/
  int    i, j, p;
  DTYPE  zero = 0.0, tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Acol(i,p) * Brow(p,j);

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

void gemm_base_ABresident( char orderA, char transM, 
                           int m, int n, int k, 
                           DTYPE alpha, DTYPE *A, int ldA, 
                                        DTYPE *B, int ldB, 
                           DTYPE beta,  DTYPE *C, int ldC ){
/*
  Baseline micro-kernel 
  Replace with specialized micro-kernel where A-->m x k is resident in registers
*/
  int    i, j, p;
  DTYPE  zero = 0.0, tmp;

  if ( transM=='N' ) {
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( orderA=='C' )
          for ( p=0; p<k; p++ ) 
            tmp += Acol(i,p) * Bcol(p,j);
        else
          for ( p=0; p<k; p++ ) 
            tmp += Arow(i,p) * Bcol(p,j);
  
        if (beta==zero)
          Ccol(i,j) = alpha*tmp;
        else
          Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
      }
  }
  else {
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( orderA=='C' )
          for ( p=0; p<k; p++ ) 
            tmp += Acol(p,i) * Bcol(p,j);
        else
          for ( p=0; p<k; p++ ) 
            tmp += Arow(p,i) * Bcol(p,j);
  
        if (beta==zero)
          Ccol(i,j) = alpha*tmp;
        else
          Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
      }
  }
}

