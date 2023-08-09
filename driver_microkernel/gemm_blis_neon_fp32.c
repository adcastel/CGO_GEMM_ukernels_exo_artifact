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
#include <arm_neon.h>

#include "blis_neon.h"
#include "gemm_blis_neon_fp32.h"

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Ctcol(a1,a2) Ctmp[ (a2)*(ldCt)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Ctrow(a1,a2) Ctmp[ (a1)*(ldCt)+(a2) ]

#define Ctref(a1,a2) Ctmp[ (a2)*(ldCt)+(a1) ]
#define Atref(a1,a2) Atmp[ (a2)*(Atlda)+(a1) ]

#define MR 8
#define NR 12


inline void gemm_microkernel_Cresident_neon_8x12_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ){

    int         i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
    float32x4_t C000, C001, C002, C003, 
		C004, C005, C006, C007, 
		C008, C009, C010, C011,
                C100, C101, C102, C103, 
		C104, C105, C106, C107, 
	        C108, C109, C110, C111,	
                A000, A001, A002, A003, 
		A004, A005, A006, A007, 
		A008, A009, A010, A011, 
                A100, A101, A102, A103, 
		A104, A105, A106, A107, 
		A108, A109, A110, A111; 
{
#define A0    A000
#define A1    A002
#define B0    A100
#define B1    A101
#define B2    A102

    float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];
    if ( kc==0 )
	    return;
    C000 = vmovq_n_f32(0);
    C001 = vmovq_n_f32(0);
    C002 = vmovq_n_f32(0);
    C003 = vmovq_n_f32(0);
    C004 = vmovq_n_f32(0);
    C005 = vmovq_n_f32(0);
    C006 = vmovq_n_f32(0);
    C007 = vmovq_n_f32(0);
    C008 = vmovq_n_f32(0);
    C009 = vmovq_n_f32(0);
    C010 = vmovq_n_f32(0);
    C011 = vmovq_n_f32(0);
    C100 = vmovq_n_f32(0);
    C101 = vmovq_n_f32(0);
    C102 = vmovq_n_f32(0);
    C103 = vmovq_n_f32(0);
    C104 = vmovq_n_f32(0);
    C105 = vmovq_n_f32(0);
    C106 = vmovq_n_f32(0);
    C107 = vmovq_n_f32(0);
    C108 = vmovq_n_f32(0);
    C109 = vmovq_n_f32(0);
    C110 = vmovq_n_f32(0);
    C111 = vmovq_n_f32(0);
    
    if ( orderC=='C' ) {
	    Aptr = &Ar[0];
	    Bptr = &Br[0];
	    Amr  = MR;
	    Bnr  = NR;
    }
    else {
	    Aptr = &Br[0];
	    Bptr = &Ar[0];
	    Amr  = NR;
	    Bnr  = MR;
    }
    if ( alpha!=zero ) {
	    for ( k=0; k<kc; k++ ) {
		    
		    A0 = vld1q_f32(&Aptr[baseA]);
		    A1 = vld1q_f32(&Aptr[baseA+4]);
		    B0 = vld1q_f32(&Bptr[baseB]);
		    B1 = vld1q_f32(&Bptr[baseB+4]);
		    B2 = vld1q_f32(&Bptr[baseB+8]);
		    
		    C000 = vfmaq_laneq_f32(C000, A0, B0, 0);
		    C001 = vfmaq_laneq_f32(C001, A0, B0, 1);
		    C002 = vfmaq_laneq_f32(C002, A0, B0, 2);
		    C003 = vfmaq_laneq_f32(C003, A0, B0, 3);
		    
		    C004 = vfmaq_laneq_f32(C004, A0, B1, 0);
		    C005 = vfmaq_laneq_f32(C005, A0, B1, 1);
		    C006 = vfmaq_laneq_f32(C006, A0, B1, 2);
		    C007 = vfmaq_laneq_f32(C007, A0, B1, 3);
		    
		    C008 = vfmaq_laneq_f32(C008, A0, B2, 0);
		    C009 = vfmaq_laneq_f32(C009, A0, B2, 1);
		    C010 = vfmaq_laneq_f32(C010, A0, B2, 2);
		    C011 = vfmaq_laneq_f32(C011, A0, B2, 3);
		 
		    C100 = vfmaq_laneq_f32(C100, A1, B0, 0);
		    C101 = vfmaq_laneq_f32(C101, A1, B0, 1);
		    C102 = vfmaq_laneq_f32(C102, A1, B0, 2);
		    C103 = vfmaq_laneq_f32(C103, A1, B0, 3);
		    
		    C104 = vfmaq_laneq_f32(C104, A1, B1, 0);
		    C105 = vfmaq_laneq_f32(C105, A1, B1, 1);
		    C106 = vfmaq_laneq_f32(C106, A1, B1, 2);
		    C107 = vfmaq_laneq_f32(C107, A1, B1, 3);
		    
		    C108 = vfmaq_laneq_f32(C108, A1, B2, 0);
		    C109 = vfmaq_laneq_f32(C109, A1, B2, 1);
		    C110 = vfmaq_laneq_f32(C110, A1, B2, 2);
		    C111 = vfmaq_laneq_f32(C111, A1, B2, 3);
		 
		    baseA = baseA+Amr; 
		    baseB = baseB+Bnr;
	    }
	    
/*	    if ( alpha==-one ) {
		    C000 = -C000; C001 = -C001; C002 = -C002; C003 = -C003; C004 = -C004; C005 = -C005; C006 = -C006; C007 = -C007; 
		    C100 = -C100; C101 = -C101; C102 = -C102; C103 = -C103; C104 = -C104; C105 = -C105; C106 = -C106; C107 = -C107; 
		    C008 = -C008; C009 = -C009; C010 = -C010; C011 = -C011;
		    C108 = -C108; C109 = -C109; C110 = -C110; C111 = -C111;
	    }*/
	    /*else if ( alpha!=one ) {
		    C00 = alpha*C00; C01 = alpha*C01; C02 = alpha*C02; C03 = alpha*C03; C04 = alpha*C04; C05 = alpha*C05; C06 = alpha*C06; C07 = alpha*C07;
		    C10 = alpha*C10; C11 = alpha*C11; C12 = alpha*C12; C13 = alpha*C13; C14 = alpha*C14; C15 = alpha*C15; C16 = alpha*C16; C17 = alpha*C17;
		    C20 = alpha*C20; C21 = alpha*C21; C22 = alpha*C22; C23 = alpha*C23; C24 = alpha*C24; C25 = alpha*C25; C26 = alpha*C26; C27 = alpha*C27;
	    }*/
    }
   
    if ( (mr<MR)||(nr<NR) ) {
	    vst1q_f32(&Ctref(0,0), C000);
	    vst1q_f32(&Ctref(0,1), C001);
	    vst1q_f32(&Ctref(0,2), C002);
	    vst1q_f32(&Ctref(0,3), C003);
	    vst1q_f32(&Ctref(0,4), C004);
	    vst1q_f32(&Ctref(0,5), C005);
	    vst1q_f32(&Ctref(0,6), C006);
	    vst1q_f32(&Ctref(0,7), C007);
	    vst1q_f32(&Ctref(0,8), C008);
	    vst1q_f32(&Ctref(0,9), C009);
	    vst1q_f32(&Ctref(0,10), C010);
	    vst1q_f32(&Ctref(0,11), C011);
	    
	    vst1q_f32(&Ctref(4,0), C100);
	    vst1q_f32(&Ctref(4,1), C101);
	    vst1q_f32(&Ctref(4,2), C102);
	    vst1q_f32(&Ctref(4,3), C103);
	    vst1q_f32(&Ctref(4,4), C104);
	    vst1q_f32(&Ctref(4,5), C105);
	    vst1q_f32(&Ctref(4,6), C106);
	    vst1q_f32(&Ctref(4,7), C107);
	    vst1q_f32(&Ctref(4,8), C108);
	    vst1q_f32(&Ctref(4,9), C109);
	    vst1q_f32(&Ctref(4,10), C110);
	    vst1q_f32(&Ctref(4,11), C111);

	    
	    if ( beta!=zero ) {
		    if ( orderC=='C' ) {
			    for ( j=0; j<nr; j++ ) 
				    for ( i=0; i<mr; i++ ) 
					    Ccol(i,j) = beta*Ccol(i,j) + Ctcol(i,j);
		    }
		    else {
			    for ( j=0; j<nr; j++ ) 
				    for ( i=0; i<mr; i++ ) 
					    Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
		    }
	    }
	    else {
		    if ( orderC=='C' ) {
			    for ( j=0; j<nr; j++ ) 
				    for ( i=0; i<mr; i++ ) 
					    Ccol(i,j) = Ctcol(i,j);
		    }
		    else {
			    for ( j=0; j<nr; j++ ) 
				    for ( i=0; i<mr; i++ ) 
					    Crow(i,j) = Ctrow(i,j);
		    }
	    }
    }
    else if ( (mr==MR)&&(nr==NR) ) {
	    if ( beta!=zero ) {
		    A000 = vld1q_f32(&Ccol(0,0));
		    A001 = vld1q_f32(&Ccol(0,1));
		    A002 = vld1q_f32(&Ccol(0,2));
		    A003 = vld1q_f32(&Ccol(0,3));
		    A004 = vld1q_f32(&Ccol(0,4));
		    A005 = vld1q_f32(&Ccol(0,5));
		    A006 = vld1q_f32(&Ccol(0,6));
		    A007 = vld1q_f32(&Ccol(0,7));
		    A008 = vld1q_f32(&Ccol(0,8));
		    A009 = vld1q_f32(&Ccol(0,9));
		    A010 = vld1q_f32(&Ccol(0,10));
		    A011 = vld1q_f32(&Ccol(0,11));
		    
		    A100 = vld1q_f32(&Ccol(4,0));
		    A101 = vld1q_f32(&Ccol(4,1));
		    A102 = vld1q_f32(&Ccol(4,2));
		    A103 = vld1q_f32(&Ccol(4,3));
		    A104 = vld1q_f32(&Ccol(4,4));
		    A105 = vld1q_f32(&Ccol(4,5));
		    A106 = vld1q_f32(&Ccol(4,6));
		    A107 = vld1q_f32(&Ccol(4,7));
		    A108 = vld1q_f32(&Ccol(4,8));
		    A109 = vld1q_f32(&Ccol(4,9));
		    A110 = vld1q_f32(&Ccol(4,10));
		    A111 = vld1q_f32(&Ccol(4,11));
		    
		    C000 = beta*A000 + C000;
		    C001 = beta*A001 + C001;
		    C002 = beta*A002 + C002;
		    C003 = beta*A003 + C003;
		    C004 = beta*A004 + C004;
		    C005 = beta*A005 + C005;
		    C006 = beta*A006 + C006;
		    C007 = beta*A007 + C007;
		    C008 = beta*A008 + C008;
		    C009 = beta*A009 + C009;
		    C010 = beta*A010 + C010;
		    C011 = beta*A011 + C011;
		 
		    C100 = beta*A100 + C100;
		    C101 = beta*A101 + C101;
		    C102 = beta*A102 + C102;
		    C103 = beta*A103 + C103;
		    C104 = beta*A104 + C104;
		    C105 = beta*A105 + C105;
		    C106 = beta*A106 + C106;
		    C107 = beta*A107 + C107;
		    C108 = beta*A108 + C108;
		    C109 = beta*A109 + C109;
		    C110 = beta*A110 + C110;
		    C111 = beta*A111 + C111;
		    
	    }
	    
	    vst1q_f32(&Ccol(0,0), C000);
	    vst1q_f32(&Ccol(0,1), C001);
	    vst1q_f32(&Ccol(0,2), C002);
	    vst1q_f32(&Ccol(0,3), C003);
	    vst1q_f32(&Ccol(0,4), C004);
	    vst1q_f32(&Ccol(0,5), C005);
	    vst1q_f32(&Ccol(0,6), C006);
	    vst1q_f32(&Ccol(0,7), C007);
	    vst1q_f32(&Ccol(0,8), C008);
	    vst1q_f32(&Ccol(0,9), C009);
	    vst1q_f32(&Ccol(0,10), C010);
	    vst1q_f32(&Ccol(0,11), C011);
	    
	    vst1q_f32(&Ccol(4,0), C100);
	    vst1q_f32(&Ccol(4,1), C101);
	    vst1q_f32(&Ccol(4,2), C102);
	    vst1q_f32(&Ccol(4,3), C103);
	    vst1q_f32(&Ccol(4,4), C104);
	    vst1q_f32(&Ccol(4,5), C105);
	    vst1q_f32(&Ccol(4,6), C106);
	    vst1q_f32(&Ccol(4,7), C107);
	    vst1q_f32(&Ccol(4,8), C108);
	    vst1q_f32(&Ccol(4,9), C109);
	    vst1q_f32(&Ccol(4,10), C110);
	    vst1q_f32(&Ccol(4,11), C111);
	    
    }
    else {
	    printf("Error: Incorrect use of 8x12 micro-kernel with %d x %d block\n", mr, nr);
	    exit(-1);
    }
}
}


inline void fvtrans_float32_4x4( float32x4_t *A0, float32x4_t *A1, float32x4_t *A2, float32x4_t *A3 ) {
  float32x4x2_t V = vtrnq_f32 ( (float32x4_t) vtrn1q_f64 ( (float64x2_t) *A0, (float64x2_t) *A2 ),
                                (float32x4_t) vtrn1q_f64 ( (float64x2_t) *A1, (float64x2_t) *A3 ));
  float32x4x2_t W = vtrnq_f32 ( (float32x4_t) vtrn2q_f64 ( (float64x2_t) *A0, (float64x2_t) *A2 ),
                                (float32x4_t) vtrn2q_f64 ( (float64x2_t) *A1, (float64x2_t) *A3 ));
  *A0 = V.val[0];
  *A1 = V.val[1];
  *A2 = W.val[0];
  *A3 = W.val[1];
}

inline void fvtrans_float32_8x8( float32x4_t *A00, float32x4_t *A01, float32x4_t *A02, float32x4_t *A03,
                                 float32x4_t *A04, float32x4_t *A05, float32x4_t *A06, float32x4_t *A07,
                                 float32x4_t *A10, float32x4_t *A11, float32x4_t *A12, float32x4_t *A13,
                                 float32x4_t *A14, float32x4_t *A15, float32x4_t *A16, float32x4_t *A17 ) {
  fvtrans_float32_4x4( A00, A01, A02, A03 );
  fvtrans_float32_4x4( A10, A11, A12, A13 );
  fvtrans_float32_4x4( A04, A05, A06, A07);
  fvtrans_float32_4x4( A14, A15, A16, A17);
}

inline float32_t dot(float32x4_t a, float32x4_t b){
	         float32_t output;
		          float32x4_t product = vmlaq_f32(product, a, b);
			           output = vaddvq_f32(product);
				            return output;
}

