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

#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "dtypes.h"

void generate_matrix( char orderM, size_t m, size_t n, DTYPE *M, size_t ldM )
{
/*
 * Generate a matrix with random entries
 * m      : Row dimension
 * n      : Column dimension
 * M      : Matrix
 * ldM   : Leading dimension
 *
 */
  int i, j;
  
  if ( orderM=='C' ) {
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
#if defined(FP16)
        Mcol(i,j) = ((((DTYPE) j*m+i)/m)/n);
#else
        Mcol(i,j) = ((DTYPE) rand())/RAND_MAX + 1.0;
        // Mcol(i,j) = ((int) (10.0 * ((DTYPE) rand())/RAND_MAX));
#endif
  }else
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
#if defined(FP16)
        Mrow(i,j) = ((((DTYPE) j*m+i)/m)/n);
#else
        Mrow(i,j) = ((DTYPE) rand())/RAND_MAX + 1.0;
        // Mrow(i,j) = ((int) (10.0 * ((DTYPE) rand())/RAND_MAX));
#endif
}
/*===========================================================================*/
void print_matrix( char *name, char orderM, size_t m, size_t n, DTYPE *M, size_t ldM )
{
/*
 * Print a matrix to standard output
 * name   : Label for matrix name
 * m      : Row dimension
 * n      : Column dimension
 * A      : Matrix
 *
 */
  size_t i, j;

  if ( orderM=='C' )
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
#if defined(FP16)
        printf( "%s[%zu,%zu] = %8.2e;\n", name, i, j, ((double) Mcol(i,j)) );
#elif defined(FP32)
        printf( "%s[%zu,%zu] = %14.8e;\n", name, i, j, ((double) Mcol(i,j)) );
#elif defined(FP64)
        printf( "%s[%zu,%zu] = %22.16e;\n", name, i, j, ((double) Mcol(i,j)) );
#endif
  else
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
#if defined(FP16)
        printf( "%s[%zu,%zu] = %8.2e;\n", name, i, j, ((double) Mrow(i,j)) );
#elif defined(FP32)
        printf( "%s[%zu,%zu] = %14.8e;\n", name, i, j, ((double) Mrow(i,j)) );
#elif defined(FP64)
        printf( "%s[%zu,%zu] = %22.16e;\n", name, i, j, ((double) Mrow(i,j)) );
#endif
}
/*===========================================================================*/
double dclock()
{
/* 
 * Timer
 *
 */
  struct timeval  tv;
  // struct timezone tz;

  gettimeofday( &tv, NULL );   

  return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}
/*===========================================================================*/
