#include "model_level.h"

/**
  Purpose
    Estimate the dimension of the panels that will fit into a given level of the cache hierarchy following the
    principles in the paper "Analytical modeling is enough for enough for high performance BLIS" by 
    T. M. Low et al, 2016

  Inputs:
     NL:    Number of sets
     CL:    Bytes per line
     WL:    Associativity degree
     (m,n): Dimensions of block in higher level of cache
     Sdata: Bytes per element (e.g., 8 for FP64)

  Output
     k:     Determines that a block of size k x n stays in this level of the cache

  Rule of thumb: subtract 1 line from WL (associativity), which is dedicated to the
  operand which does not reside in the cache, and distribute the rest between the two 
  other operands proportionaly to the ratio n/m   
  For example, with the conventional algorithm B3A2B1C0 and the L1 cache, 
  1 line is dedicated to Cr (non-resident in cache) while the remaining lines are distributed
  between Ar (mr x kc) and Br (kc x nr) proportionally to the ratio nr/mr to estimate kc
**/

int model_level(int isL3, int NL, int CL, int WL, int dataSize, int m, int n) { 

  
  int k, CAr, CBr;

  if (WL==2) {
     if (!isL3)
       k = NL * CL / (2.0 * m * dataSize);
     else  
       k = NL * CL / (2.0 * n * dataSize);
  } else {
     CAr = floor( ( (float)WL - 1.0 ) / (1.0 + (float)n / (float)m) ); //Lines of each set for Ar 
     if (CAr==0) { // Special case
       CAr = 1.0;
       CBr = WL - 2;
       if (!isL3)
         k = CBr * NL * CL / (n * dataSize);
       else   
         k = CBr * NL * CL / (m * dataSize);
     } else {
       CBr = ceil( ( (float)n / (float)m ) * (float)CAr ); //Lines of each set for Br
       if (!isL3)
         k = CAr * NL * CL / (m * dataSize);
       else
         k = CBr * NL * CL / (n * dataSize);
     }
  }

   return k;

}

void get_optim_mc_nc_kc(int dataSize, int m, int n, int k, int mr, int nr, int *mc, int *nc, int *kc) {
  
  *kc = model_level(0, NL1, CL1, WL1, dataSize, mr, nr); *kc = floor(*kc);
  *kc = min(k, *kc);
  *mc = model_level(0, NL2, CL2, WL2, dataSize, *kc, nr); *mc = floor(*mc);
  *mc = min(m, *mc);
  *nc = model_level(1, NL3, CL3, WL3, dataSize, *kc, *mc); *nc = floor(*nc);
  *nc = min(n, *nc);
}

