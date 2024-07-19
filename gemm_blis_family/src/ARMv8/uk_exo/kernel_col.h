
#pragma once
#ifndef KERNEL_COL_H
#define KERNEL_COL_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_2f32{
    float * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f32c{
    const float * const data;
    const int_fast32_t strides[2];
};
// gemm_NEON_10x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 10] @DRAM
// )
void gemm_NEON_10x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 10] @DRAM
// )
void gemm_NEON_10x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 10] @DRAM
// )
void gemm_NEON_10x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 10] @DRAM
// )
void gemm_NEON_10x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 10] @DRAM
// )
void gemm_NEON_10x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 10] @DRAM
// )
void gemm_NEON_10x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 10] @DRAM
// )
void gemm_NEON_10x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 10] @DRAM
// )
void gemm_NEON_10x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 10] @DRAM
// )
void gemm_NEON_10x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 10] @DRAM
// )
void gemm_NEON_10x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 10] @DRAM
// )
void gemm_NEON_10x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 10] @DRAM
// )
void gemm_NEON_10x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 10] @DRAM
// )
void gemm_NEON_10x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 10] @DRAM
// )
void gemm_NEON_10x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 10] @DRAM
// )
void gemm_NEON_10x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 10] @DRAM
// )
void gemm_NEON_10x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 10] @DRAM
// )
void gemm_NEON_10x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 10] @DRAM
// )
void gemm_NEON_10x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 10] @DRAM
// )
void gemm_NEON_10x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 10] @DRAM
// )
void gemm_NEON_10x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 10] @DRAM
// )
void gemm_NEON_10x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 10] @DRAM
// )
void gemm_NEON_10x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 10] @DRAM
// )
void gemm_NEON_10x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 10] @DRAM
// )
void gemm_NEON_10x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 10] @DRAM
// )
void gemm_NEON_10x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 10] @DRAM
// )
void gemm_NEON_10x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 10] @DRAM
// )
void gemm_NEON_10x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 10] @DRAM
// )
void gemm_NEON_10x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 10] @DRAM
// )
void gemm_NEON_10x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 10] @DRAM
// )
void gemm_NEON_10x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 10] @DRAM
// )
void gemm_NEON_10x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 10] @DRAM
// )
void gemm_NEON_10x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 10] @DRAM
// )
void gemm_NEON_10x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 10] @DRAM
// )
void gemm_NEON_10x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 10] @DRAM
// )
void gemm_NEON_10x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 10] @DRAM
// )
void gemm_NEON_10x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 10] @DRAM
// )
void gemm_NEON_10x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 10] @DRAM
// )
void gemm_NEON_10x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 10] @DRAM
// )
void gemm_NEON_10x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 10] @DRAM
// )
void gemm_NEON_10x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 10] @DRAM
// )
void gemm_NEON_10x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 10] @DRAM
// )
void gemm_NEON_10x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 10] @DRAM
// )
void gemm_NEON_10x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 10] @DRAM
// )
void gemm_NEON_10x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 10] @DRAM
// )
void gemm_NEON_10x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 10] @DRAM
// )
void gemm_NEON_10x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 10] @DRAM
// )
void gemm_NEON_10x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_10x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 10] @DRAM
// )
void gemm_NEON_10x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 11] @DRAM
// )
void gemm_NEON_11x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 11] @DRAM
// )
void gemm_NEON_11x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 11] @DRAM
// )
void gemm_NEON_11x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 11] @DRAM
// )
void gemm_NEON_11x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 11] @DRAM
// )
void gemm_NEON_11x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 11] @DRAM
// )
void gemm_NEON_11x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 11] @DRAM
// )
void gemm_NEON_11x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 11] @DRAM
// )
void gemm_NEON_11x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 11] @DRAM
// )
void gemm_NEON_11x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 11] @DRAM
// )
void gemm_NEON_11x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 11] @DRAM
// )
void gemm_NEON_11x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 11] @DRAM
// )
void gemm_NEON_11x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 11] @DRAM
// )
void gemm_NEON_11x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 11] @DRAM
// )
void gemm_NEON_11x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 11] @DRAM
// )
void gemm_NEON_11x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 11] @DRAM
// )
void gemm_NEON_11x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 11] @DRAM
// )
void gemm_NEON_11x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 11] @DRAM
// )
void gemm_NEON_11x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 11] @DRAM
// )
void gemm_NEON_11x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 11] @DRAM
// )
void gemm_NEON_11x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 11] @DRAM
// )
void gemm_NEON_11x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 11] @DRAM
// )
void gemm_NEON_11x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 11] @DRAM
// )
void gemm_NEON_11x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 11] @DRAM
// )
void gemm_NEON_11x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 11] @DRAM
// )
void gemm_NEON_11x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 11] @DRAM
// )
void gemm_NEON_11x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 11] @DRAM
// )
void gemm_NEON_11x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 11] @DRAM
// )
void gemm_NEON_11x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 11] @DRAM
// )
void gemm_NEON_11x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 11] @DRAM
// )
void gemm_NEON_11x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 11] @DRAM
// )
void gemm_NEON_11x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 11] @DRAM
// )
void gemm_NEON_11x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 11] @DRAM
// )
void gemm_NEON_11x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 11] @DRAM
// )
void gemm_NEON_11x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 11] @DRAM
// )
void gemm_NEON_11x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 11] @DRAM
// )
void gemm_NEON_11x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 11] @DRAM
// )
void gemm_NEON_11x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 11] @DRAM
// )
void gemm_NEON_11x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 11] @DRAM
// )
void gemm_NEON_11x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 11] @DRAM
// )
void gemm_NEON_11x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 11] @DRAM
// )
void gemm_NEON_11x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 11] @DRAM
// )
void gemm_NEON_11x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 11] @DRAM
// )
void gemm_NEON_11x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 11] @DRAM
// )
void gemm_NEON_11x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 11] @DRAM
// )
void gemm_NEON_11x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 11] @DRAM
// )
void gemm_NEON_11x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 11] @DRAM
// )
void gemm_NEON_11x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_11x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 11] @DRAM
// )
void gemm_NEON_11x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 12] @DRAM
// )
void gemm_NEON_12x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 12] @DRAM
// )
void gemm_NEON_12x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 12] @DRAM
// )
void gemm_NEON_12x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 12] @DRAM
// )
void gemm_NEON_12x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 12] @DRAM
// )
void gemm_NEON_12x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 12] @DRAM
// )
void gemm_NEON_12x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 12] @DRAM
// )
void gemm_NEON_12x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 12] @DRAM
// )
void gemm_NEON_12x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 12] @DRAM
// )
void gemm_NEON_12x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 12] @DRAM
// )
void gemm_NEON_12x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 12] @DRAM
// )
void gemm_NEON_12x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 12] @DRAM
// )
void gemm_NEON_12x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 12] @DRAM
// )
void gemm_NEON_12x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 12] @DRAM
// )
void gemm_NEON_12x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 12] @DRAM
// )
void gemm_NEON_12x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 12] @DRAM
// )
void gemm_NEON_12x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 12] @DRAM
// )
void gemm_NEON_12x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 12] @DRAM
// )
void gemm_NEON_12x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 12] @DRAM
// )
void gemm_NEON_12x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 12] @DRAM
// )
void gemm_NEON_12x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 12] @DRAM
// )
void gemm_NEON_12x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 12] @DRAM
// )
void gemm_NEON_12x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 12] @DRAM
// )
void gemm_NEON_12x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 12] @DRAM
// )
void gemm_NEON_12x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 12] @DRAM
// )
void gemm_NEON_12x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 12] @DRAM
// )
void gemm_NEON_12x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 12] @DRAM
// )
void gemm_NEON_12x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 12] @DRAM
// )
void gemm_NEON_12x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 12] @DRAM
// )
void gemm_NEON_12x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 12] @DRAM
// )
void gemm_NEON_12x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 12] @DRAM
// )
void gemm_NEON_12x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 12] @DRAM
// )
void gemm_NEON_12x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 12] @DRAM
// )
void gemm_NEON_12x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 12] @DRAM
// )
void gemm_NEON_12x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 12] @DRAM
// )
void gemm_NEON_12x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 12] @DRAM
// )
void gemm_NEON_12x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_NEON_12x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_NEON_12x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 12] @DRAM
// )
void gemm_NEON_12x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 12] @DRAM
// )
void gemm_NEON_12x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 12] @DRAM
// )
void gemm_NEON_12x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 12] @DRAM
// )
void gemm_NEON_12x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 12] @DRAM
// )
void gemm_NEON_12x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 12] @DRAM
// )
void gemm_NEON_12x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 12] @DRAM
// )
void gemm_NEON_12x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 12] @DRAM
// )
void gemm_NEON_12x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 12] @DRAM
// )
void gemm_NEON_12x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_12x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 12] @DRAM
// )
void gemm_NEON_12x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 13] @DRAM
// )
void gemm_NEON_13x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 13] @DRAM
// )
void gemm_NEON_13x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 13] @DRAM
// )
void gemm_NEON_13x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 13] @DRAM
// )
void gemm_NEON_13x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 13] @DRAM
// )
void gemm_NEON_13x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 13] @DRAM
// )
void gemm_NEON_13x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 13] @DRAM
// )
void gemm_NEON_13x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 13] @DRAM
// )
void gemm_NEON_13x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 13] @DRAM
// )
void gemm_NEON_13x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 13] @DRAM
// )
void gemm_NEON_13x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 13] @DRAM
// )
void gemm_NEON_13x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 13] @DRAM
// )
void gemm_NEON_13x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 13] @DRAM
// )
void gemm_NEON_13x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 13] @DRAM
// )
void gemm_NEON_13x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 13] @DRAM
// )
void gemm_NEON_13x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 13] @DRAM
// )
void gemm_NEON_13x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 13] @DRAM
// )
void gemm_NEON_13x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 13] @DRAM
// )
void gemm_NEON_13x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 13] @DRAM
// )
void gemm_NEON_13x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 13] @DRAM
// )
void gemm_NEON_13x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 13] @DRAM
// )
void gemm_NEON_13x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 13] @DRAM
// )
void gemm_NEON_13x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 13] @DRAM
// )
void gemm_NEON_13x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 13] @DRAM
// )
void gemm_NEON_13x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 13] @DRAM
// )
void gemm_NEON_13x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 13] @DRAM
// )
void gemm_NEON_13x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 13] @DRAM
// )
void gemm_NEON_13x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 13] @DRAM
// )
void gemm_NEON_13x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 13] @DRAM
// )
void gemm_NEON_13x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 13] @DRAM
// )
void gemm_NEON_13x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 13] @DRAM
// )
void gemm_NEON_13x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 13] @DRAM
// )
void gemm_NEON_13x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 13] @DRAM
// )
void gemm_NEON_13x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 13] @DRAM
// )
void gemm_NEON_13x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 13] @DRAM
// )
void gemm_NEON_13x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 13] @DRAM
// )
void gemm_NEON_13x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 13] @DRAM
// )
void gemm_NEON_13x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 13] @DRAM
// )
void gemm_NEON_13x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 13] @DRAM
// )
void gemm_NEON_13x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 13] @DRAM
// )
void gemm_NEON_13x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 13] @DRAM
// )
void gemm_NEON_13x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 13] @DRAM
// )
void gemm_NEON_13x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 13] @DRAM
// )
void gemm_NEON_13x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 13] @DRAM
// )
void gemm_NEON_13x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 13] @DRAM
// )
void gemm_NEON_13x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 13] @DRAM
// )
void gemm_NEON_13x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 13] @DRAM
// )
void gemm_NEON_13x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_13x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 13] @DRAM
// )
void gemm_NEON_13x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 14] @DRAM
// )
void gemm_NEON_14x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 14] @DRAM
// )
void gemm_NEON_14x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 14] @DRAM
// )
void gemm_NEON_14x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 14] @DRAM
// )
void gemm_NEON_14x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 14] @DRAM
// )
void gemm_NEON_14x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 14] @DRAM
// )
void gemm_NEON_14x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 14] @DRAM
// )
void gemm_NEON_14x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 14] @DRAM
// )
void gemm_NEON_14x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 14] @DRAM
// )
void gemm_NEON_14x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 14] @DRAM
// )
void gemm_NEON_14x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 14] @DRAM
// )
void gemm_NEON_14x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 14] @DRAM
// )
void gemm_NEON_14x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 14] @DRAM
// )
void gemm_NEON_14x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 14] @DRAM
// )
void gemm_NEON_14x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 14] @DRAM
// )
void gemm_NEON_14x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 14] @DRAM
// )
void gemm_NEON_14x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 14] @DRAM
// )
void gemm_NEON_14x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 14] @DRAM
// )
void gemm_NEON_14x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 14] @DRAM
// )
void gemm_NEON_14x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 14] @DRAM
// )
void gemm_NEON_14x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 14] @DRAM
// )
void gemm_NEON_14x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 14] @DRAM
// )
void gemm_NEON_14x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 14] @DRAM
// )
void gemm_NEON_14x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 14] @DRAM
// )
void gemm_NEON_14x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 14] @DRAM
// )
void gemm_NEON_14x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 14] @DRAM
// )
void gemm_NEON_14x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 14] @DRAM
// )
void gemm_NEON_14x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 14] @DRAM
// )
void gemm_NEON_14x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 14] @DRAM
// )
void gemm_NEON_14x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 14] @DRAM
// )
void gemm_NEON_14x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 14] @DRAM
// )
void gemm_NEON_14x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 14] @DRAM
// )
void gemm_NEON_14x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 14] @DRAM
// )
void gemm_NEON_14x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 14] @DRAM
// )
void gemm_NEON_14x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 14] @DRAM
// )
void gemm_NEON_14x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 14] @DRAM
// )
void gemm_NEON_14x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 14] @DRAM
// )
void gemm_NEON_14x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 14] @DRAM
// )
void gemm_NEON_14x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 14] @DRAM
// )
void gemm_NEON_14x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 14] @DRAM
// )
void gemm_NEON_14x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 14] @DRAM
// )
void gemm_NEON_14x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 14] @DRAM
// )
void gemm_NEON_14x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 14] @DRAM
// )
void gemm_NEON_14x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 14] @DRAM
// )
void gemm_NEON_14x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 14] @DRAM
// )
void gemm_NEON_14x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 14] @DRAM
// )
void gemm_NEON_14x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 14] @DRAM
// )
void gemm_NEON_14x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_14x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 14] @DRAM
// )
void gemm_NEON_14x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 15] @DRAM
// )
void gemm_NEON_15x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 15] @DRAM
// )
void gemm_NEON_15x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 15] @DRAM
// )
void gemm_NEON_15x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 15] @DRAM
// )
void gemm_NEON_15x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 15] @DRAM
// )
void gemm_NEON_15x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 15] @DRAM
// )
void gemm_NEON_15x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 15] @DRAM
// )
void gemm_NEON_15x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 15] @DRAM
// )
void gemm_NEON_15x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 15] @DRAM
// )
void gemm_NEON_15x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 15] @DRAM
// )
void gemm_NEON_15x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 15] @DRAM
// )
void gemm_NEON_15x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 15] @DRAM
// )
void gemm_NEON_15x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 15] @DRAM
// )
void gemm_NEON_15x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 15] @DRAM
// )
void gemm_NEON_15x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 15] @DRAM
// )
void gemm_NEON_15x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 15] @DRAM
// )
void gemm_NEON_15x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 15] @DRAM
// )
void gemm_NEON_15x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 15] @DRAM
// )
void gemm_NEON_15x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 15] @DRAM
// )
void gemm_NEON_15x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 15] @DRAM
// )
void gemm_NEON_15x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 15] @DRAM
// )
void gemm_NEON_15x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 15] @DRAM
// )
void gemm_NEON_15x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 15] @DRAM
// )
void gemm_NEON_15x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 15] @DRAM
// )
void gemm_NEON_15x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 15] @DRAM
// )
void gemm_NEON_15x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 15] @DRAM
// )
void gemm_NEON_15x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 15] @DRAM
// )
void gemm_NEON_15x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 15] @DRAM
// )
void gemm_NEON_15x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 15] @DRAM
// )
void gemm_NEON_15x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 15] @DRAM
// )
void gemm_NEON_15x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 15] @DRAM
// )
void gemm_NEON_15x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 15] @DRAM
// )
void gemm_NEON_15x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 15] @DRAM
// )
void gemm_NEON_15x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 15] @DRAM
// )
void gemm_NEON_15x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 15] @DRAM
// )
void gemm_NEON_15x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 15] @DRAM
// )
void gemm_NEON_15x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 15] @DRAM
// )
void gemm_NEON_15x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 15] @DRAM
// )
void gemm_NEON_15x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 15] @DRAM
// )
void gemm_NEON_15x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 15] @DRAM
// )
void gemm_NEON_15x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 15] @DRAM
// )
void gemm_NEON_15x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 15] @DRAM
// )
void gemm_NEON_15x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 15] @DRAM
// )
void gemm_NEON_15x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 15] @DRAM
// )
void gemm_NEON_15x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 15] @DRAM
// )
void gemm_NEON_15x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 15] @DRAM
// )
void gemm_NEON_15x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 15] @DRAM
// )
void gemm_NEON_15x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_15x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 15] @DRAM
// )
void gemm_NEON_15x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 16] @DRAM
// )
void gemm_NEON_16x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 16] @DRAM
// )
void gemm_NEON_16x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 16] @DRAM
// )
void gemm_NEON_16x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 16] @DRAM
// )
void gemm_NEON_16x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 16] @DRAM
// )
void gemm_NEON_16x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 16] @DRAM
// )
void gemm_NEON_16x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 16] @DRAM
// )
void gemm_NEON_16x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 16] @DRAM
// )
void gemm_NEON_16x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 16] @DRAM
// )
void gemm_NEON_16x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 16] @DRAM
// )
void gemm_NEON_16x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 16] @DRAM
// )
void gemm_NEON_16x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 16] @DRAM
// )
void gemm_NEON_16x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 16] @DRAM
// )
void gemm_NEON_16x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 16] @DRAM
// )
void gemm_NEON_16x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 16] @DRAM
// )
void gemm_NEON_16x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 16] @DRAM
// )
void gemm_NEON_16x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 16] @DRAM
// )
void gemm_NEON_16x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 16] @DRAM
// )
void gemm_NEON_16x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 16] @DRAM
// )
void gemm_NEON_16x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 16] @DRAM
// )
void gemm_NEON_16x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 16] @DRAM
// )
void gemm_NEON_16x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 16] @DRAM
// )
void gemm_NEON_16x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 16] @DRAM
// )
void gemm_NEON_16x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 16] @DRAM
// )
void gemm_NEON_16x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 16] @DRAM
// )
void gemm_NEON_16x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 16] @DRAM
// )
void gemm_NEON_16x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 16] @DRAM
// )
void gemm_NEON_16x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 16] @DRAM
// )
void gemm_NEON_16x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 16] @DRAM
// )
void gemm_NEON_16x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 16] @DRAM
// )
void gemm_NEON_16x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 16] @DRAM
// )
void gemm_NEON_16x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 16] @DRAM
// )
void gemm_NEON_16x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 16] @DRAM
// )
void gemm_NEON_16x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 16] @DRAM
// )
void gemm_NEON_16x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 16] @DRAM
// )
void gemm_NEON_16x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 16] @DRAM
// )
void gemm_NEON_16x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_NEON_16x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_NEON_16x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 16] @DRAM
// )
void gemm_NEON_16x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 16] @DRAM
// )
void gemm_NEON_16x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 16] @DRAM
// )
void gemm_NEON_16x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 16] @DRAM
// )
void gemm_NEON_16x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 16] @DRAM
// )
void gemm_NEON_16x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 16] @DRAM
// )
void gemm_NEON_16x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 16] @DRAM
// )
void gemm_NEON_16x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 16] @DRAM
// )
void gemm_NEON_16x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 16] @DRAM
// )
void gemm_NEON_16x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_16x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 16] @DRAM
// )
void gemm_NEON_16x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 17] @DRAM
// )
void gemm_NEON_17x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 17] @DRAM
// )
void gemm_NEON_17x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 17] @DRAM
// )
void gemm_NEON_17x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 17] @DRAM
// )
void gemm_NEON_17x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 17] @DRAM
// )
void gemm_NEON_17x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 17] @DRAM
// )
void gemm_NEON_17x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 17] @DRAM
// )
void gemm_NEON_17x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 17] @DRAM
// )
void gemm_NEON_17x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 17] @DRAM
// )
void gemm_NEON_17x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 17] @DRAM
// )
void gemm_NEON_17x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 17] @DRAM
// )
void gemm_NEON_17x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 17] @DRAM
// )
void gemm_NEON_17x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 17] @DRAM
// )
void gemm_NEON_17x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 17] @DRAM
// )
void gemm_NEON_17x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 17] @DRAM
// )
void gemm_NEON_17x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 17] @DRAM
// )
void gemm_NEON_17x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 17] @DRAM
// )
void gemm_NEON_17x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 17] @DRAM
// )
void gemm_NEON_17x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 17] @DRAM
// )
void gemm_NEON_17x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 17] @DRAM
// )
void gemm_NEON_17x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 17] @DRAM
// )
void gemm_NEON_17x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 17] @DRAM
// )
void gemm_NEON_17x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 17] @DRAM
// )
void gemm_NEON_17x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 17] @DRAM
// )
void gemm_NEON_17x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 17] @DRAM
// )
void gemm_NEON_17x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 17] @DRAM
// )
void gemm_NEON_17x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 17] @DRAM
// )
void gemm_NEON_17x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 17] @DRAM
// )
void gemm_NEON_17x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 17] @DRAM
// )
void gemm_NEON_17x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 17] @DRAM
// )
void gemm_NEON_17x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 17] @DRAM
// )
void gemm_NEON_17x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 17] @DRAM
// )
void gemm_NEON_17x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 17] @DRAM
// )
void gemm_NEON_17x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 17] @DRAM
// )
void gemm_NEON_17x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 17] @DRAM
// )
void gemm_NEON_17x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 17] @DRAM
// )
void gemm_NEON_17x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 17] @DRAM
// )
void gemm_NEON_17x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 17] @DRAM
// )
void gemm_NEON_17x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 17] @DRAM
// )
void gemm_NEON_17x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 17] @DRAM
// )
void gemm_NEON_17x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 17] @DRAM
// )
void gemm_NEON_17x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 17] @DRAM
// )
void gemm_NEON_17x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 17] @DRAM
// )
void gemm_NEON_17x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 17] @DRAM
// )
void gemm_NEON_17x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 17] @DRAM
// )
void gemm_NEON_17x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 17] @DRAM
// )
void gemm_NEON_17x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 17] @DRAM
// )
void gemm_NEON_17x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_17x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 17] @DRAM
// )
void gemm_NEON_17x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 18] @DRAM
// )
void gemm_NEON_18x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 18] @DRAM
// )
void gemm_NEON_18x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 18] @DRAM
// )
void gemm_NEON_18x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 18] @DRAM
// )
void gemm_NEON_18x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 18] @DRAM
// )
void gemm_NEON_18x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 18] @DRAM
// )
void gemm_NEON_18x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 18] @DRAM
// )
void gemm_NEON_18x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 18] @DRAM
// )
void gemm_NEON_18x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 18] @DRAM
// )
void gemm_NEON_18x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 18] @DRAM
// )
void gemm_NEON_18x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 18] @DRAM
// )
void gemm_NEON_18x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 18] @DRAM
// )
void gemm_NEON_18x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 18] @DRAM
// )
void gemm_NEON_18x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 18] @DRAM
// )
void gemm_NEON_18x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 18] @DRAM
// )
void gemm_NEON_18x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 18] @DRAM
// )
void gemm_NEON_18x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 18] @DRAM
// )
void gemm_NEON_18x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 18] @DRAM
// )
void gemm_NEON_18x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 18] @DRAM
// )
void gemm_NEON_18x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 18] @DRAM
// )
void gemm_NEON_18x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 18] @DRAM
// )
void gemm_NEON_18x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 18] @DRAM
// )
void gemm_NEON_18x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 18] @DRAM
// )
void gemm_NEON_18x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 18] @DRAM
// )
void gemm_NEON_18x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 18] @DRAM
// )
void gemm_NEON_18x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 18] @DRAM
// )
void gemm_NEON_18x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 18] @DRAM
// )
void gemm_NEON_18x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 18] @DRAM
// )
void gemm_NEON_18x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 18] @DRAM
// )
void gemm_NEON_18x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 18] @DRAM
// )
void gemm_NEON_18x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 18] @DRAM
// )
void gemm_NEON_18x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 18] @DRAM
// )
void gemm_NEON_18x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 18] @DRAM
// )
void gemm_NEON_18x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 18] @DRAM
// )
void gemm_NEON_18x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 18] @DRAM
// )
void gemm_NEON_18x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 18] @DRAM
// )
void gemm_NEON_18x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 18] @DRAM
// )
void gemm_NEON_18x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 18] @DRAM
// )
void gemm_NEON_18x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 18] @DRAM
// )
void gemm_NEON_18x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 18] @DRAM
// )
void gemm_NEON_18x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 18] @DRAM
// )
void gemm_NEON_18x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 18] @DRAM
// )
void gemm_NEON_18x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 18] @DRAM
// )
void gemm_NEON_18x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 18] @DRAM
// )
void gemm_NEON_18x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 18] @DRAM
// )
void gemm_NEON_18x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 18] @DRAM
// )
void gemm_NEON_18x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 18] @DRAM
// )
void gemm_NEON_18x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_18x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 18] @DRAM
// )
void gemm_NEON_18x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 19] @DRAM
// )
void gemm_NEON_19x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 19] @DRAM
// )
void gemm_NEON_19x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 19] @DRAM
// )
void gemm_NEON_19x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 19] @DRAM
// )
void gemm_NEON_19x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 19] @DRAM
// )
void gemm_NEON_19x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 19] @DRAM
// )
void gemm_NEON_19x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 19] @DRAM
// )
void gemm_NEON_19x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 19] @DRAM
// )
void gemm_NEON_19x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 19] @DRAM
// )
void gemm_NEON_19x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 19] @DRAM
// )
void gemm_NEON_19x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 19] @DRAM
// )
void gemm_NEON_19x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 19] @DRAM
// )
void gemm_NEON_19x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 19] @DRAM
// )
void gemm_NEON_19x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 19] @DRAM
// )
void gemm_NEON_19x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 19] @DRAM
// )
void gemm_NEON_19x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 19] @DRAM
// )
void gemm_NEON_19x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 19] @DRAM
// )
void gemm_NEON_19x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 19] @DRAM
// )
void gemm_NEON_19x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 19] @DRAM
// )
void gemm_NEON_19x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 19] @DRAM
// )
void gemm_NEON_19x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 19] @DRAM
// )
void gemm_NEON_19x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 19] @DRAM
// )
void gemm_NEON_19x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 19] @DRAM
// )
void gemm_NEON_19x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 19] @DRAM
// )
void gemm_NEON_19x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 19] @DRAM
// )
void gemm_NEON_19x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 19] @DRAM
// )
void gemm_NEON_19x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 19] @DRAM
// )
void gemm_NEON_19x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 19] @DRAM
// )
void gemm_NEON_19x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 19] @DRAM
// )
void gemm_NEON_19x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 19] @DRAM
// )
void gemm_NEON_19x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 19] @DRAM
// )
void gemm_NEON_19x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 19] @DRAM
// )
void gemm_NEON_19x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 19] @DRAM
// )
void gemm_NEON_19x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 19] @DRAM
// )
void gemm_NEON_19x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 19] @DRAM
// )
void gemm_NEON_19x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 19] @DRAM
// )
void gemm_NEON_19x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 19] @DRAM
// )
void gemm_NEON_19x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 19] @DRAM
// )
void gemm_NEON_19x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 19] @DRAM
// )
void gemm_NEON_19x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 19] @DRAM
// )
void gemm_NEON_19x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 19] @DRAM
// )
void gemm_NEON_19x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 19] @DRAM
// )
void gemm_NEON_19x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 19] @DRAM
// )
void gemm_NEON_19x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 19] @DRAM
// )
void gemm_NEON_19x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 19] @DRAM
// )
void gemm_NEON_19x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 19] @DRAM
// )
void gemm_NEON_19x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 19] @DRAM
// )
void gemm_NEON_19x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_19x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 19] @DRAM
// )
void gemm_NEON_19x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 1] @DRAM
// )
void gemm_NEON_1x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 1] @DRAM
// )
void gemm_NEON_1x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 1] @DRAM
// )
void gemm_NEON_1x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 1] @DRAM
// )
void gemm_NEON_1x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 1] @DRAM
// )
void gemm_NEON_1x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 1] @DRAM
// )
void gemm_NEON_1x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 1] @DRAM
// )
void gemm_NEON_1x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 1] @DRAM
// )
void gemm_NEON_1x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 1] @DRAM
// )
void gemm_NEON_1x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 1] @DRAM
// )
void gemm_NEON_1x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 1] @DRAM
// )
void gemm_NEON_1x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 1] @DRAM
// )
void gemm_NEON_1x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 1] @DRAM
// )
void gemm_NEON_1x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 1] @DRAM
// )
void gemm_NEON_1x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_1x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_1x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 1] @DRAM
// )
void gemm_NEON_1x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 1] @DRAM
// )
void gemm_NEON_1x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 1] @DRAM
// )
void gemm_NEON_1x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 1] @DRAM
// )
void gemm_NEON_1x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 1] @DRAM
// )
void gemm_NEON_1x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 1] @DRAM
// )
void gemm_NEON_1x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 1] @DRAM
// )
void gemm_NEON_1x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 1] @DRAM
// )
void gemm_NEON_1x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 1] @DRAM
// )
void gemm_NEON_1x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 1] @DRAM
// )
void gemm_NEON_1x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_1x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_1x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_1x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_1x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_1x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_1x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_1x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 20] @DRAM
// )
void gemm_NEON_20x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 20] @DRAM
// )
void gemm_NEON_20x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 20] @DRAM
// )
void gemm_NEON_20x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 20] @DRAM
// )
void gemm_NEON_20x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 20] @DRAM
// )
void gemm_NEON_20x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 20] @DRAM
// )
void gemm_NEON_20x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 20] @DRAM
// )
void gemm_NEON_20x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 20] @DRAM
// )
void gemm_NEON_20x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 20] @DRAM
// )
void gemm_NEON_20x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 20] @DRAM
// )
void gemm_NEON_20x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 20] @DRAM
// )
void gemm_NEON_20x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 20] @DRAM
// )
void gemm_NEON_20x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 20] @DRAM
// )
void gemm_NEON_20x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 20] @DRAM
// )
void gemm_NEON_20x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 20] @DRAM
// )
void gemm_NEON_20x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 20] @DRAM
// )
void gemm_NEON_20x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 20] @DRAM
// )
void gemm_NEON_20x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 20] @DRAM
// )
void gemm_NEON_20x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 20] @DRAM
// )
void gemm_NEON_20x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 20] @DRAM
// )
void gemm_NEON_20x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 20] @DRAM
// )
void gemm_NEON_20x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 20] @DRAM
// )
void gemm_NEON_20x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 20] @DRAM
// )
void gemm_NEON_20x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 20] @DRAM
// )
void gemm_NEON_20x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 20] @DRAM
// )
void gemm_NEON_20x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 20] @DRAM
// )
void gemm_NEON_20x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 20] @DRAM
// )
void gemm_NEON_20x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 20] @DRAM
// )
void gemm_NEON_20x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 20] @DRAM
// )
void gemm_NEON_20x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 20] @DRAM
// )
void gemm_NEON_20x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 20] @DRAM
// )
void gemm_NEON_20x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 20] @DRAM
// )
void gemm_NEON_20x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 20] @DRAM
// )
void gemm_NEON_20x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 20] @DRAM
// )
void gemm_NEON_20x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 20] @DRAM
// )
void gemm_NEON_20x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 20] @DRAM
// )
void gemm_NEON_20x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_NEON_20x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_NEON_20x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 20] @DRAM
// )
void gemm_NEON_20x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 20] @DRAM
// )
void gemm_NEON_20x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 20] @DRAM
// )
void gemm_NEON_20x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 20] @DRAM
// )
void gemm_NEON_20x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 20] @DRAM
// )
void gemm_NEON_20x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 20] @DRAM
// )
void gemm_NEON_20x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 20] @DRAM
// )
void gemm_NEON_20x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 20] @DRAM
// )
void gemm_NEON_20x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 20] @DRAM
// )
void gemm_NEON_20x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_20x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 20] @DRAM
// )
void gemm_NEON_20x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 21] @DRAM
// )
void gemm_NEON_21x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 21] @DRAM
// )
void gemm_NEON_21x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 21] @DRAM
// )
void gemm_NEON_21x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 21] @DRAM
// )
void gemm_NEON_21x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 21] @DRAM
// )
void gemm_NEON_21x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 21] @DRAM
// )
void gemm_NEON_21x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 21] @DRAM
// )
void gemm_NEON_21x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 21] @DRAM
// )
void gemm_NEON_21x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 21] @DRAM
// )
void gemm_NEON_21x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 21] @DRAM
// )
void gemm_NEON_21x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 21] @DRAM
// )
void gemm_NEON_21x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 21] @DRAM
// )
void gemm_NEON_21x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 21] @DRAM
// )
void gemm_NEON_21x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 21] @DRAM
// )
void gemm_NEON_21x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 21] @DRAM
// )
void gemm_NEON_21x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 21] @DRAM
// )
void gemm_NEON_21x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 21] @DRAM
// )
void gemm_NEON_21x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 21] @DRAM
// )
void gemm_NEON_21x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 21] @DRAM
// )
void gemm_NEON_21x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 21] @DRAM
// )
void gemm_NEON_21x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 21] @DRAM
// )
void gemm_NEON_21x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 21] @DRAM
// )
void gemm_NEON_21x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 21] @DRAM
// )
void gemm_NEON_21x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 21] @DRAM
// )
void gemm_NEON_21x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 21] @DRAM
// )
void gemm_NEON_21x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 21] @DRAM
// )
void gemm_NEON_21x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 21] @DRAM
// )
void gemm_NEON_21x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 21] @DRAM
// )
void gemm_NEON_21x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 21] @DRAM
// )
void gemm_NEON_21x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 21] @DRAM
// )
void gemm_NEON_21x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 21] @DRAM
// )
void gemm_NEON_21x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 21] @DRAM
// )
void gemm_NEON_21x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 21] @DRAM
// )
void gemm_NEON_21x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 21] @DRAM
// )
void gemm_NEON_21x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 21] @DRAM
// )
void gemm_NEON_21x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 21] @DRAM
// )
void gemm_NEON_21x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 21] @DRAM
// )
void gemm_NEON_21x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 21] @DRAM
// )
void gemm_NEON_21x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 21] @DRAM
// )
void gemm_NEON_21x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 21] @DRAM
// )
void gemm_NEON_21x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 21] @DRAM
// )
void gemm_NEON_21x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 21] @DRAM
// )
void gemm_NEON_21x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 21] @DRAM
// )
void gemm_NEON_21x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 21] @DRAM
// )
void gemm_NEON_21x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 21] @DRAM
// )
void gemm_NEON_21x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 21] @DRAM
// )
void gemm_NEON_21x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 21] @DRAM
// )
void gemm_NEON_21x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_21x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 21] @DRAM
// )
void gemm_NEON_21x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 22] @DRAM
// )
void gemm_NEON_22x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 22] @DRAM
// )
void gemm_NEON_22x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 22] @DRAM
// )
void gemm_NEON_22x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 22] @DRAM
// )
void gemm_NEON_22x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 22] @DRAM
// )
void gemm_NEON_22x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 22] @DRAM
// )
void gemm_NEON_22x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 22] @DRAM
// )
void gemm_NEON_22x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 22] @DRAM
// )
void gemm_NEON_22x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 22] @DRAM
// )
void gemm_NEON_22x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 22] @DRAM
// )
void gemm_NEON_22x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 22] @DRAM
// )
void gemm_NEON_22x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 22] @DRAM
// )
void gemm_NEON_22x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 22] @DRAM
// )
void gemm_NEON_22x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 22] @DRAM
// )
void gemm_NEON_22x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 22] @DRAM
// )
void gemm_NEON_22x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 22] @DRAM
// )
void gemm_NEON_22x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 22] @DRAM
// )
void gemm_NEON_22x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 22] @DRAM
// )
void gemm_NEON_22x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 22] @DRAM
// )
void gemm_NEON_22x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 22] @DRAM
// )
void gemm_NEON_22x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 22] @DRAM
// )
void gemm_NEON_22x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 22] @DRAM
// )
void gemm_NEON_22x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 22] @DRAM
// )
void gemm_NEON_22x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 22] @DRAM
// )
void gemm_NEON_22x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 22] @DRAM
// )
void gemm_NEON_22x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 22] @DRAM
// )
void gemm_NEON_22x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 22] @DRAM
// )
void gemm_NEON_22x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 22] @DRAM
// )
void gemm_NEON_22x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 22] @DRAM
// )
void gemm_NEON_22x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 22] @DRAM
// )
void gemm_NEON_22x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 22] @DRAM
// )
void gemm_NEON_22x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 22] @DRAM
// )
void gemm_NEON_22x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 22] @DRAM
// )
void gemm_NEON_22x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 22] @DRAM
// )
void gemm_NEON_22x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 22] @DRAM
// )
void gemm_NEON_22x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 22] @DRAM
// )
void gemm_NEON_22x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 22] @DRAM
// )
void gemm_NEON_22x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 22] @DRAM
// )
void gemm_NEON_22x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 22] @DRAM
// )
void gemm_NEON_22x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 22] @DRAM
// )
void gemm_NEON_22x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 22] @DRAM
// )
void gemm_NEON_22x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 22] @DRAM
// )
void gemm_NEON_22x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 22] @DRAM
// )
void gemm_NEON_22x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 22] @DRAM
// )
void gemm_NEON_22x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 22] @DRAM
// )
void gemm_NEON_22x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 22] @DRAM
// )
void gemm_NEON_22x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 22] @DRAM
// )
void gemm_NEON_22x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_22x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 22] @DRAM
// )
void gemm_NEON_22x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 23] @DRAM
// )
void gemm_NEON_23x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 23] @DRAM
// )
void gemm_NEON_23x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 23] @DRAM
// )
void gemm_NEON_23x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 23] @DRAM
// )
void gemm_NEON_23x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 23] @DRAM
// )
void gemm_NEON_23x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 23] @DRAM
// )
void gemm_NEON_23x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 23] @DRAM
// )
void gemm_NEON_23x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 23] @DRAM
// )
void gemm_NEON_23x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 23] @DRAM
// )
void gemm_NEON_23x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 23] @DRAM
// )
void gemm_NEON_23x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 23] @DRAM
// )
void gemm_NEON_23x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 23] @DRAM
// )
void gemm_NEON_23x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 23] @DRAM
// )
void gemm_NEON_23x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 23] @DRAM
// )
void gemm_NEON_23x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 23] @DRAM
// )
void gemm_NEON_23x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 23] @DRAM
// )
void gemm_NEON_23x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 23] @DRAM
// )
void gemm_NEON_23x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 23] @DRAM
// )
void gemm_NEON_23x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 23] @DRAM
// )
void gemm_NEON_23x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 23] @DRAM
// )
void gemm_NEON_23x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 23] @DRAM
// )
void gemm_NEON_23x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 23] @DRAM
// )
void gemm_NEON_23x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 23] @DRAM
// )
void gemm_NEON_23x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 23] @DRAM
// )
void gemm_NEON_23x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 23] @DRAM
// )
void gemm_NEON_23x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 23] @DRAM
// )
void gemm_NEON_23x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 23] @DRAM
// )
void gemm_NEON_23x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 23] @DRAM
// )
void gemm_NEON_23x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 23] @DRAM
// )
void gemm_NEON_23x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 23] @DRAM
// )
void gemm_NEON_23x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 23] @DRAM
// )
void gemm_NEON_23x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 23] @DRAM
// )
void gemm_NEON_23x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 23] @DRAM
// )
void gemm_NEON_23x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 23] @DRAM
// )
void gemm_NEON_23x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 23] @DRAM
// )
void gemm_NEON_23x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 23] @DRAM
// )
void gemm_NEON_23x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 23] @DRAM
// )
void gemm_NEON_23x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 23] @DRAM
// )
void gemm_NEON_23x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 23] @DRAM
// )
void gemm_NEON_23x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 23] @DRAM
// )
void gemm_NEON_23x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 23] @DRAM
// )
void gemm_NEON_23x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 23] @DRAM
// )
void gemm_NEON_23x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 23] @DRAM
// )
void gemm_NEON_23x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 23] @DRAM
// )
void gemm_NEON_23x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 23] @DRAM
// )
void gemm_NEON_23x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 23] @DRAM
// )
void gemm_NEON_23x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 23] @DRAM
// )
void gemm_NEON_23x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_23x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 23] @DRAM
// )
void gemm_NEON_23x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 24] @DRAM
// )
void gemm_NEON_24x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 24] @DRAM
// )
void gemm_NEON_24x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 24] @DRAM
// )
void gemm_NEON_24x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 24] @DRAM
// )
void gemm_NEON_24x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 24] @DRAM
// )
void gemm_NEON_24x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 24] @DRAM
// )
void gemm_NEON_24x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 24] @DRAM
// )
void gemm_NEON_24x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 24] @DRAM
// )
void gemm_NEON_24x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 24] @DRAM
// )
void gemm_NEON_24x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 24] @DRAM
// )
void gemm_NEON_24x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 24] @DRAM
// )
void gemm_NEON_24x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 24] @DRAM
// )
void gemm_NEON_24x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 24] @DRAM
// )
void gemm_NEON_24x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 24] @DRAM
// )
void gemm_NEON_24x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 24] @DRAM
// )
void gemm_NEON_24x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 24] @DRAM
// )
void gemm_NEON_24x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 24] @DRAM
// )
void gemm_NEON_24x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 24] @DRAM
// )
void gemm_NEON_24x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 24] @DRAM
// )
void gemm_NEON_24x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 24] @DRAM
// )
void gemm_NEON_24x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 24] @DRAM
// )
void gemm_NEON_24x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 24] @DRAM
// )
void gemm_NEON_24x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 24] @DRAM
// )
void gemm_NEON_24x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 24] @DRAM
// )
void gemm_NEON_24x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 24] @DRAM
// )
void gemm_NEON_24x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 24] @DRAM
// )
void gemm_NEON_24x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 24] @DRAM
// )
void gemm_NEON_24x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 24] @DRAM
// )
void gemm_NEON_24x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 24] @DRAM
// )
void gemm_NEON_24x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 24] @DRAM
// )
void gemm_NEON_24x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 24] @DRAM
// )
void gemm_NEON_24x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 24] @DRAM
// )
void gemm_NEON_24x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 24] @DRAM
// )
void gemm_NEON_24x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 24] @DRAM
// )
void gemm_NEON_24x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 24] @DRAM
// )
void gemm_NEON_24x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 24] @DRAM
// )
void gemm_NEON_24x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 24] @DRAM
// )
void gemm_NEON_24x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 24] @DRAM
// )
void gemm_NEON_24x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 24] @DRAM
// )
void gemm_NEON_24x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 24] @DRAM
// )
void gemm_NEON_24x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 24] @DRAM
// )
void gemm_NEON_24x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 24] @DRAM
// )
void gemm_NEON_24x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 24] @DRAM
// )
void gemm_NEON_24x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 24] @DRAM
// )
void gemm_NEON_24x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 24] @DRAM
// )
void gemm_NEON_24x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 24] @DRAM
// )
void gemm_NEON_24x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 24] @DRAM
// )
void gemm_NEON_24x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_24x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 24] @DRAM
// )
void gemm_NEON_24x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 2] @DRAM
// )
void gemm_NEON_2x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 2] @DRAM
// )
void gemm_NEON_2x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 2] @DRAM
// )
void gemm_NEON_2x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 2] @DRAM
// )
void gemm_NEON_2x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 2] @DRAM
// )
void gemm_NEON_2x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 2] @DRAM
// )
void gemm_NEON_2x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 2] @DRAM
// )
void gemm_NEON_2x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 2] @DRAM
// )
void gemm_NEON_2x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 2] @DRAM
// )
void gemm_NEON_2x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 2] @DRAM
// )
void gemm_NEON_2x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 2] @DRAM
// )
void gemm_NEON_2x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 2] @DRAM
// )
void gemm_NEON_2x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 2] @DRAM
// )
void gemm_NEON_2x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 2] @DRAM
// )
void gemm_NEON_2x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_2x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_2x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 2] @DRAM
// )
void gemm_NEON_2x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 2] @DRAM
// )
void gemm_NEON_2x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 2] @DRAM
// )
void gemm_NEON_2x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 2] @DRAM
// )
void gemm_NEON_2x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 2] @DRAM
// )
void gemm_NEON_2x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 2] @DRAM
// )
void gemm_NEON_2x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 2] @DRAM
// )
void gemm_NEON_2x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 2] @DRAM
// )
void gemm_NEON_2x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 2] @DRAM
// )
void gemm_NEON_2x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 2] @DRAM
// )
void gemm_NEON_2x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_2x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_2x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_2x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_2x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_2x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_2x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_2x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 3] @DRAM
// )
void gemm_NEON_3x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 3] @DRAM
// )
void gemm_NEON_3x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 3] @DRAM
// )
void gemm_NEON_3x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 3] @DRAM
// )
void gemm_NEON_3x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 3] @DRAM
// )
void gemm_NEON_3x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 3] @DRAM
// )
void gemm_NEON_3x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 3] @DRAM
// )
void gemm_NEON_3x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 3] @DRAM
// )
void gemm_NEON_3x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 3] @DRAM
// )
void gemm_NEON_3x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 3] @DRAM
// )
void gemm_NEON_3x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 3] @DRAM
// )
void gemm_NEON_3x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 3] @DRAM
// )
void gemm_NEON_3x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 3] @DRAM
// )
void gemm_NEON_3x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 3] @DRAM
// )
void gemm_NEON_3x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_3x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_3x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 3] @DRAM
// )
void gemm_NEON_3x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 3] @DRAM
// )
void gemm_NEON_3x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 3] @DRAM
// )
void gemm_NEON_3x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 3] @DRAM
// )
void gemm_NEON_3x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 3] @DRAM
// )
void gemm_NEON_3x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 3] @DRAM
// )
void gemm_NEON_3x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 3] @DRAM
// )
void gemm_NEON_3x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 3] @DRAM
// )
void gemm_NEON_3x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 3] @DRAM
// )
void gemm_NEON_3x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 3] @DRAM
// )
void gemm_NEON_3x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_3x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_3x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_3x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_3x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_3x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_3x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_3x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 4] @DRAM
// )
void gemm_NEON_4x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 4] @DRAM
// )
void gemm_NEON_4x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 4] @DRAM
// )
void gemm_NEON_4x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 4] @DRAM
// )
void gemm_NEON_4x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 4] @DRAM
// )
void gemm_NEON_4x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 4] @DRAM
// )
void gemm_NEON_4x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_NEON_4x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_NEON_4x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 4] @DRAM
// )
void gemm_NEON_4x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 4] @DRAM
// )
void gemm_NEON_4x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 4] @DRAM
// )
void gemm_NEON_4x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 4] @DRAM
// )
void gemm_NEON_4x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 4] @DRAM
// )
void gemm_NEON_4x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 4] @DRAM
// )
void gemm_NEON_4x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 4] @DRAM
// )
void gemm_NEON_4x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 4] @DRAM
// )
void gemm_NEON_4x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 4] @DRAM
// )
void gemm_NEON_4x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 4] @DRAM
// )
void gemm_NEON_4x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 4] @DRAM
// )
void gemm_NEON_4x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 4] @DRAM
// )
void gemm_NEON_4x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 4] @DRAM
// )
void gemm_NEON_4x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 4] @DRAM
// )
void gemm_NEON_4x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 4] @DRAM
// )
void gemm_NEON_4x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 4] @DRAM
// )
void gemm_NEON_4x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_4x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 5] @DRAM
// )
void gemm_NEON_5x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 5] @DRAM
// )
void gemm_NEON_5x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 5] @DRAM
// )
void gemm_NEON_5x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 5] @DRAM
// )
void gemm_NEON_5x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 5] @DRAM
// )
void gemm_NEON_5x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 5] @DRAM
// )
void gemm_NEON_5x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 5] @DRAM
// )
void gemm_NEON_5x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 5] @DRAM
// )
void gemm_NEON_5x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 5] @DRAM
// )
void gemm_NEON_5x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 5] @DRAM
// )
void gemm_NEON_5x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 5] @DRAM
// )
void gemm_NEON_5x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 5] @DRAM
// )
void gemm_NEON_5x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 5] @DRAM
// )
void gemm_NEON_5x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 5] @DRAM
// )
void gemm_NEON_5x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 5] @DRAM
// )
void gemm_NEON_5x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 5] @DRAM
// )
void gemm_NEON_5x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 5] @DRAM
// )
void gemm_NEON_5x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 5] @DRAM
// )
void gemm_NEON_5x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 5] @DRAM
// )
void gemm_NEON_5x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 5] @DRAM
// )
void gemm_NEON_5x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 5] @DRAM
// )
void gemm_NEON_5x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 5] @DRAM
// )
void gemm_NEON_5x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 5] @DRAM
// )
void gemm_NEON_5x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 5] @DRAM
// )
void gemm_NEON_5x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 5] @DRAM
// )
void gemm_NEON_5x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 5] @DRAM
// )
void gemm_NEON_5x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 5] @DRAM
// )
void gemm_NEON_5x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 5] @DRAM
// )
void gemm_NEON_5x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 5] @DRAM
// )
void gemm_NEON_5x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 5] @DRAM
// )
void gemm_NEON_5x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 5] @DRAM
// )
void gemm_NEON_5x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 5] @DRAM
// )
void gemm_NEON_5x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 5] @DRAM
// )
void gemm_NEON_5x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 5] @DRAM
// )
void gemm_NEON_5x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 5] @DRAM
// )
void gemm_NEON_5x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 5] @DRAM
// )
void gemm_NEON_5x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 5] @DRAM
// )
void gemm_NEON_5x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 5] @DRAM
// )
void gemm_NEON_5x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 5] @DRAM
// )
void gemm_NEON_5x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 5] @DRAM
// )
void gemm_NEON_5x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 5] @DRAM
// )
void gemm_NEON_5x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 5] @DRAM
// )
void gemm_NEON_5x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 5] @DRAM
// )
void gemm_NEON_5x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 5] @DRAM
// )
void gemm_NEON_5x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 5] @DRAM
// )
void gemm_NEON_5x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 5] @DRAM
// )
void gemm_NEON_5x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 5] @DRAM
// )
void gemm_NEON_5x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_5x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 5] @DRAM
// )
void gemm_NEON_5x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 6] @DRAM
// )
void gemm_NEON_6x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 6] @DRAM
// )
void gemm_NEON_6x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 6] @DRAM
// )
void gemm_NEON_6x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 6] @DRAM
// )
void gemm_NEON_6x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 6] @DRAM
// )
void gemm_NEON_6x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 6] @DRAM
// )
void gemm_NEON_6x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 6] @DRAM
// )
void gemm_NEON_6x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 6] @DRAM
// )
void gemm_NEON_6x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 6] @DRAM
// )
void gemm_NEON_6x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 6] @DRAM
// )
void gemm_NEON_6x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 6] @DRAM
// )
void gemm_NEON_6x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 6] @DRAM
// )
void gemm_NEON_6x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 6] @DRAM
// )
void gemm_NEON_6x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 6] @DRAM
// )
void gemm_NEON_6x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 6] @DRAM
// )
void gemm_NEON_6x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 6] @DRAM
// )
void gemm_NEON_6x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 6] @DRAM
// )
void gemm_NEON_6x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 6] @DRAM
// )
void gemm_NEON_6x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 6] @DRAM
// )
void gemm_NEON_6x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 6] @DRAM
// )
void gemm_NEON_6x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 6] @DRAM
// )
void gemm_NEON_6x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 6] @DRAM
// )
void gemm_NEON_6x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 6] @DRAM
// )
void gemm_NEON_6x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 6] @DRAM
// )
void gemm_NEON_6x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 6] @DRAM
// )
void gemm_NEON_6x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 6] @DRAM
// )
void gemm_NEON_6x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 6] @DRAM
// )
void gemm_NEON_6x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 6] @DRAM
// )
void gemm_NEON_6x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 6] @DRAM
// )
void gemm_NEON_6x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 6] @DRAM
// )
void gemm_NEON_6x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 6] @DRAM
// )
void gemm_NEON_6x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 6] @DRAM
// )
void gemm_NEON_6x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 6] @DRAM
// )
void gemm_NEON_6x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 6] @DRAM
// )
void gemm_NEON_6x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 6] @DRAM
// )
void gemm_NEON_6x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 6] @DRAM
// )
void gemm_NEON_6x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 6] @DRAM
// )
void gemm_NEON_6x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 6] @DRAM
// )
void gemm_NEON_6x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 6] @DRAM
// )
void gemm_NEON_6x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 6] @DRAM
// )
void gemm_NEON_6x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 6] @DRAM
// )
void gemm_NEON_6x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 6] @DRAM
// )
void gemm_NEON_6x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 6] @DRAM
// )
void gemm_NEON_6x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 6] @DRAM
// )
void gemm_NEON_6x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 6] @DRAM
// )
void gemm_NEON_6x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 6] @DRAM
// )
void gemm_NEON_6x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 6] @DRAM
// )
void gemm_NEON_6x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_6x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 6] @DRAM
// )
void gemm_NEON_6x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 7] @DRAM
// )
void gemm_NEON_7x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 7] @DRAM
// )
void gemm_NEON_7x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 7] @DRAM
// )
void gemm_NEON_7x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 7] @DRAM
// )
void gemm_NEON_7x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 7] @DRAM
// )
void gemm_NEON_7x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 7] @DRAM
// )
void gemm_NEON_7x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 7] @DRAM
// )
void gemm_NEON_7x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 7] @DRAM
// )
void gemm_NEON_7x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 7] @DRAM
// )
void gemm_NEON_7x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 7] @DRAM
// )
void gemm_NEON_7x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 7] @DRAM
// )
void gemm_NEON_7x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 7] @DRAM
// )
void gemm_NEON_7x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 7] @DRAM
// )
void gemm_NEON_7x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 7] @DRAM
// )
void gemm_NEON_7x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 7] @DRAM
// )
void gemm_NEON_7x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 7] @DRAM
// )
void gemm_NEON_7x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 7] @DRAM
// )
void gemm_NEON_7x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 7] @DRAM
// )
void gemm_NEON_7x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 7] @DRAM
// )
void gemm_NEON_7x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 7] @DRAM
// )
void gemm_NEON_7x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 7] @DRAM
// )
void gemm_NEON_7x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 7] @DRAM
// )
void gemm_NEON_7x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 7] @DRAM
// )
void gemm_NEON_7x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 7] @DRAM
// )
void gemm_NEON_7x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 7] @DRAM
// )
void gemm_NEON_7x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 7] @DRAM
// )
void gemm_NEON_7x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 7] @DRAM
// )
void gemm_NEON_7x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 7] @DRAM
// )
void gemm_NEON_7x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 7] @DRAM
// )
void gemm_NEON_7x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 7] @DRAM
// )
void gemm_NEON_7x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 7] @DRAM
// )
void gemm_NEON_7x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 7] @DRAM
// )
void gemm_NEON_7x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 7] @DRAM
// )
void gemm_NEON_7x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 7] @DRAM
// )
void gemm_NEON_7x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 7] @DRAM
// )
void gemm_NEON_7x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 7] @DRAM
// )
void gemm_NEON_7x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 7] @DRAM
// )
void gemm_NEON_7x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 7] @DRAM
// )
void gemm_NEON_7x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 7] @DRAM
// )
void gemm_NEON_7x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 7] @DRAM
// )
void gemm_NEON_7x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 7] @DRAM
// )
void gemm_NEON_7x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 7] @DRAM
// )
void gemm_NEON_7x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 7] @DRAM
// )
void gemm_NEON_7x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 7] @DRAM
// )
void gemm_NEON_7x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 7] @DRAM
// )
void gemm_NEON_7x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 7] @DRAM
// )
void gemm_NEON_7x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 7] @DRAM
// )
void gemm_NEON_7x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_7x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 7] @DRAM
// )
void gemm_NEON_7x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 8] @DRAM
// )
void gemm_NEON_8x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 8] @DRAM
// )
void gemm_NEON_8x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 8] @DRAM
// )
void gemm_NEON_8x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 8] @DRAM
// )
void gemm_NEON_8x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_NEON_8x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_NEON_8x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 8] @DRAM
// )
void gemm_NEON_8x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 8] @DRAM
// )
void gemm_NEON_8x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 8] @DRAM
// )
void gemm_NEON_8x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 8] @DRAM
// )
void gemm_NEON_8x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 8] @DRAM
// )
void gemm_NEON_8x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 8] @DRAM
// )
void gemm_NEON_8x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 8] @DRAM
// )
void gemm_NEON_8x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 8] @DRAM
// )
void gemm_NEON_8x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 8] @DRAM
// )
void gemm_NEON_8x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 8] @DRAM
// )
void gemm_NEON_8x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 8] @DRAM
// )
void gemm_NEON_8x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 8] @DRAM
// )
void gemm_NEON_8x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 8] @DRAM
// )
void gemm_NEON_8x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 8] @DRAM
// )
void gemm_NEON_8x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 8] @DRAM
// )
void gemm_NEON_8x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 8] @DRAM
// )
void gemm_NEON_8x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 8] @DRAM
// )
void gemm_NEON_8x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 8] @DRAM
// )
void gemm_NEON_8x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 8] @DRAM
// )
void gemm_NEON_8x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 8] @DRAM
// )
void gemm_NEON_8x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 8] @DRAM
// )
void gemm_NEON_8x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 8] @DRAM
// )
void gemm_NEON_8x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 8] @DRAM
// )
void gemm_NEON_8x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 8] @DRAM
// )
void gemm_NEON_8x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 8] @DRAM
// )
void gemm_NEON_8x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 8] @DRAM
// )
void gemm_NEON_8x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 8] @DRAM
// )
void gemm_NEON_8x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 8] @DRAM
// )
void gemm_NEON_8x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 8] @DRAM
// )
void gemm_NEON_8x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 8] @DRAM
// )
void gemm_NEON_8x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_NEON_8x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_NEON_8x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 8] @DRAM
// )
void gemm_NEON_8x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 8] @DRAM
// )
void gemm_NEON_8x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 8] @DRAM
// )
void gemm_NEON_8x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 8] @DRAM
// )
void gemm_NEON_8x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 8] @DRAM
// )
void gemm_NEON_8x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 8] @DRAM
// )
void gemm_NEON_8x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_NEON_8x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_NEON_8x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 8] @DRAM
// )
void gemm_NEON_8x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_8x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 8] @DRAM
// )
void gemm_NEON_8x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x10_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 9] @DRAM
// )
void gemm_NEON_9x10_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x10_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 9] @DRAM
// )
void gemm_NEON_9x10_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x11_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 9] @DRAM
// )
void gemm_NEON_9x11_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x11_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 9] @DRAM
// )
void gemm_NEON_9x11_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x12_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 9] @DRAM
// )
void gemm_NEON_9x12_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x12_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 9] @DRAM
// )
void gemm_NEON_9x12_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x13_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 9] @DRAM
// )
void gemm_NEON_9x13_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x13_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 9] @DRAM
// )
void gemm_NEON_9x13_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x14_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 9] @DRAM
// )
void gemm_NEON_9x14_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x14_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 9] @DRAM
// )
void gemm_NEON_9x14_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x15_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 9] @DRAM
// )
void gemm_NEON_9x15_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x15_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 9] @DRAM
// )
void gemm_NEON_9x15_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x16_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 9] @DRAM
// )
void gemm_NEON_9x16_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x16_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 9] @DRAM
// )
void gemm_NEON_9x16_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x17_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 9] @DRAM
// )
void gemm_NEON_9x17_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x17_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 9] @DRAM
// )
void gemm_NEON_9x17_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x18_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 9] @DRAM
// )
void gemm_NEON_9x18_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x18_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 9] @DRAM
// )
void gemm_NEON_9x18_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x19_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 9] @DRAM
// )
void gemm_NEON_9x19_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x19_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 9] @DRAM
// )
void gemm_NEON_9x19_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x1_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 9] @DRAM
// )
void gemm_NEON_9x1_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x1_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 9] @DRAM
// )
void gemm_NEON_9x1_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x20_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 9] @DRAM
// )
void gemm_NEON_9x20_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x20_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 9] @DRAM
// )
void gemm_NEON_9x20_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x21_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 9] @DRAM
// )
void gemm_NEON_9x21_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x21_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][21, 9] @DRAM
// )
void gemm_NEON_9x21_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x22_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 9] @DRAM
// )
void gemm_NEON_9x22_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x22_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][22, 9] @DRAM
// )
void gemm_NEON_9x22_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x23_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 9] @DRAM
// )
void gemm_NEON_9x23_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x23_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][23, 9] @DRAM
// )
void gemm_NEON_9x23_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x24_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 9] @DRAM
// )
void gemm_NEON_9x24_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x24_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 24] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][24, 9] @DRAM
// )
void gemm_NEON_9x24_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x2_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 9] @DRAM
// )
void gemm_NEON_9x2_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x2_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 9] @DRAM
// )
void gemm_NEON_9x2_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x3_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 9] @DRAM
// )
void gemm_NEON_9x3_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x3_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 9] @DRAM
// )
void gemm_NEON_9x3_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x4_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 9] @DRAM
// )
void gemm_NEON_9x4_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x4_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 9] @DRAM
// )
void gemm_NEON_9x4_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x5_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 9] @DRAM
// )
void gemm_NEON_9x5_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x5_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 9] @DRAM
// )
void gemm_NEON_9x5_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x6_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 9] @DRAM
// )
void gemm_NEON_9x6_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x6_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 9] @DRAM
// )
void gemm_NEON_9x6_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x7_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 9] @DRAM
// )
void gemm_NEON_9x7_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x7_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 9] @DRAM
// )
void gemm_NEON_9x7_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x8_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 9] @DRAM
// )
void gemm_NEON_9x8_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x8_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 9] @DRAM
// )
void gemm_NEON_9x8_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x9_b0_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 9] @DRAM
// )
void gemm_NEON_9x9_b0_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );

// gemm_NEON_9x9_b1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 9] @DRAM
// )
void gemm_NEON_9x9_b1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* b, struct exo_win_2f32 Ci );



#ifdef __cplusplus
}
#endif
#endif  // KERNEL_COL_H
