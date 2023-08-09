#include "uk_exo.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>



/* relying on the following instruction..."
neon_vfmla_4xf32_4xf32(dst,lhs,rhs,lane)
{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {lane});
*/

/* relying on the following instruction..."
neon_vld_4xf32(dst,src)
{dst_data} = vld1q_f32(&{src_data});
*/

/* relying on the following instruction..."
neon_vst_4xf32(dst,src)
vst1q_f32(&{dst_data}, {src_data});
*/
// uk_4x12_a1True_b1True(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void uk_4x12_a1True_b1True( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg[12][1];
C_reg[0][0] = vld1q_f32(&C.data[0]);
C_reg[1][0] = vld1q_f32(&C.data[C.strides[0]]);
C_reg[2][0] = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg[3][0] = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg[4][0] = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg[5][0] = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg[6][0] = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg[7][0] = vld1q_f32(&C.data[(7) * (C.strides[0])]);
C_reg[8][0] = vld1q_f32(&C.data[(8) * (C.strides[0])]);
C_reg[9][0] = vld1q_f32(&C.data[(9) * (C.strides[0])]);
C_reg[10][0] = vld1q_f32(&C.data[(10) * (C.strides[0])]);
C_reg[11][0] = vld1q_f32(&C.data[(11) * (C.strides[0])]);
float32x4_t A_reg[1];
float32x4_t B_reg[3];
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg[0] = vld1q_f32(&A[(k) * 4]);
  B_reg[0] = vld1q_f32(&B[(k) * (12)]);
  B_reg[1] = vld1q_f32(&B[(k) * (12) + 4]);
  B_reg[2] = vld1q_f32(&B[(k) * (12) + 8]);
  C_reg[0][0] = vfmaq_laneq_f32(C_reg[0][0], A_reg[0], B_reg[0], (0));
  C_reg[1][0] = vfmaq_laneq_f32(C_reg[1][0], A_reg[0], B_reg[0], (1));
  C_reg[2][0] = vfmaq_laneq_f32(C_reg[2][0], A_reg[0], B_reg[0], (2));
  C_reg[3][0] = vfmaq_laneq_f32(C_reg[3][0], A_reg[0], B_reg[0], (3));
  C_reg[4][0] = vfmaq_laneq_f32(C_reg[4][0], A_reg[0], B_reg[1], (0));
  C_reg[5][0] = vfmaq_laneq_f32(C_reg[5][0], A_reg[0], B_reg[1], (1));
  C_reg[6][0] = vfmaq_laneq_f32(C_reg[6][0], A_reg[0], B_reg[1], (2));
  C_reg[7][0] = vfmaq_laneq_f32(C_reg[7][0], A_reg[0], B_reg[1], (3));
  C_reg[8][0] = vfmaq_laneq_f32(C_reg[8][0], A_reg[0], B_reg[2], (0));
  C_reg[9][0] = vfmaq_laneq_f32(C_reg[9][0], A_reg[0], B_reg[2], (1));
  C_reg[10][0] = vfmaq_laneq_f32(C_reg[10][0], A_reg[0], B_reg[2], (2));
  C_reg[11][0] = vfmaq_laneq_f32(C_reg[11][0], A_reg[0], B_reg[2], (3));
}
vst1q_f32(&C.data[0], C_reg[0][0]);
vst1q_f32(&C.data[C.strides[0]], C_reg[1][0]);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg[2][0]);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg[3][0]);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg[4][0]);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg[5][0]);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg[6][0]);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg[7][0]);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg[8][0]);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg[9][0]);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg[10][0]);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg[11][0]);
}

// uk_4x4_a1True_b1True(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void uk_4x4_a1True_b1True( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg[4][1];
C_reg[0][0] = vld1q_f32(&C.data[0]);
C_reg[1][0] = vld1q_f32(&C.data[C.strides[0]]);
C_reg[2][0] = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg[3][0] = vld1q_f32(&C.data[(3) * (C.strides[0])]);
float32x4_t A_reg[1];
float32x4_t B_reg[1];
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg[0] = vld1q_f32(&A[(k) * 4]);
  B_reg[0] = vld1q_f32(&B[(k) * 4]);
  C_reg[0][0] = vfmaq_laneq_f32(C_reg[0][0], A_reg[0], B_reg[0], (0));
  C_reg[1][0] = vfmaq_laneq_f32(C_reg[1][0], A_reg[0], B_reg[0], (1));
  C_reg[2][0] = vfmaq_laneq_f32(C_reg[2][0], A_reg[0], B_reg[0], (2));
  C_reg[3][0] = vfmaq_laneq_f32(C_reg[3][0], A_reg[0], B_reg[0], (3));
}
vst1q_f32(&C.data[0], C_reg[0][0]);
vst1q_f32(&C.data[C.strides[0]], C_reg[1][0]);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg[2][0]);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg[3][0]);
}

// uk_4x8_a1True_b1True(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void uk_4x8_a1True_b1True( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg[8][1];
C_reg[0][0] = vld1q_f32(&C.data[0]);
C_reg[1][0] = vld1q_f32(&C.data[C.strides[0]]);
C_reg[2][0] = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg[3][0] = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg[4][0] = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg[5][0] = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg[6][0] = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg[7][0] = vld1q_f32(&C.data[(7) * (C.strides[0])]);
float32x4_t A_reg[1];
float32x4_t B_reg[2];
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg[0] = vld1q_f32(&A[(k) * 4]);
  B_reg[0] = vld1q_f32(&B[(k) * 8]);
  B_reg[1] = vld1q_f32(&B[(k) * 8 + 4]);
  C_reg[0][0] = vfmaq_laneq_f32(C_reg[0][0], A_reg[0], B_reg[0], (0));
  C_reg[1][0] = vfmaq_laneq_f32(C_reg[1][0], A_reg[0], B_reg[0], (1));
  C_reg[2][0] = vfmaq_laneq_f32(C_reg[2][0], A_reg[0], B_reg[0], (2));
  C_reg[3][0] = vfmaq_laneq_f32(C_reg[3][0], A_reg[0], B_reg[0], (3));
  C_reg[4][0] = vfmaq_laneq_f32(C_reg[4][0], A_reg[0], B_reg[1], (0));
  C_reg[5][0] = vfmaq_laneq_f32(C_reg[5][0], A_reg[0], B_reg[1], (1));
  C_reg[6][0] = vfmaq_laneq_f32(C_reg[6][0], A_reg[0], B_reg[1], (2));
  C_reg[7][0] = vfmaq_laneq_f32(C_reg[7][0], A_reg[0], B_reg[1], (3));
}
vst1q_f32(&C.data[0], C_reg[0][0]);
vst1q_f32(&C.data[C.strides[0]], C_reg[1][0]);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg[2][0]);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg[3][0]);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg[4][0]);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg[5][0]);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg[6][0]);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg[7][0]);
}

// uk_8x12_a1True_b1True(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void uk_8x12_a1True_b1True( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg[12][2];
C_reg[0][0] = vld1q_f32(&C.data[0]);
C_reg[0][1] = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg[1][0] = vld1q_f32(&C.data[C.strides[0]]);
C_reg[1][1] = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg[2][0] = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg[2][1] = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[3][0] = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg[3][1] = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[4][0] = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg[4][1] = vld1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[5][0] = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg[5][1] = vld1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[6][0] = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg[6][1] = vld1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[7][0] = vld1q_f32(&C.data[(7) * (C.strides[0])]);
C_reg[7][1] = vld1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[8][0] = vld1q_f32(&C.data[(8) * (C.strides[0])]);
C_reg[8][1] = vld1q_f32(&C.data[(8) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[9][0] = vld1q_f32(&C.data[(9) * (C.strides[0])]);
C_reg[9][1] = vld1q_f32(&C.data[(9) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[10][0] = vld1q_f32(&C.data[(10) * (C.strides[0])]);
C_reg[10][1] = vld1q_f32(&C.data[(10) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[11][0] = vld1q_f32(&C.data[(11) * (C.strides[0])]);
C_reg[11][1] = vld1q_f32(&C.data[(11) * (C.strides[0]) + (4) * (C.strides[1])]);
float32x4_t A_reg[2];
float32x4_t B_reg[3];
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg[0] = vld1q_f32(&A[(k) * 8]);
  A_reg[1] = vld1q_f32(&A[(k) * 8 + 4]);
  B_reg[0] = vld1q_f32(&B[(k) * (12)]);
  B_reg[1] = vld1q_f32(&B[(k) * (12) + 4]);
  B_reg[2] = vld1q_f32(&B[(k) * (12) + 8]);
  C_reg[0][0] = vfmaq_laneq_f32(C_reg[0][0], A_reg[0], B_reg[0], (0));
  C_reg[1][0] = vfmaq_laneq_f32(C_reg[1][0], A_reg[0], B_reg[0], (1));
  C_reg[2][0] = vfmaq_laneq_f32(C_reg[2][0], A_reg[0], B_reg[0], (2));
  C_reg[3][0] = vfmaq_laneq_f32(C_reg[3][0], A_reg[0], B_reg[0], (3));
  C_reg[0][1] = vfmaq_laneq_f32(C_reg[0][1], A_reg[1], B_reg[0], (0));
  C_reg[1][1] = vfmaq_laneq_f32(C_reg[1][1], A_reg[1], B_reg[0], (1));
  C_reg[2][1] = vfmaq_laneq_f32(C_reg[2][1], A_reg[1], B_reg[0], (2));
  C_reg[3][1] = vfmaq_laneq_f32(C_reg[3][1], A_reg[1], B_reg[0], (3));
  C_reg[4][0] = vfmaq_laneq_f32(C_reg[4][0], A_reg[0], B_reg[1], (0));
  C_reg[5][0] = vfmaq_laneq_f32(C_reg[5][0], A_reg[0], B_reg[1], (1));
  C_reg[6][0] = vfmaq_laneq_f32(C_reg[6][0], A_reg[0], B_reg[1], (2));
  C_reg[7][0] = vfmaq_laneq_f32(C_reg[7][0], A_reg[0], B_reg[1], (3));
  C_reg[4][1] = vfmaq_laneq_f32(C_reg[4][1], A_reg[1], B_reg[1], (0));
  C_reg[5][1] = vfmaq_laneq_f32(C_reg[5][1], A_reg[1], B_reg[1], (1));
  C_reg[6][1] = vfmaq_laneq_f32(C_reg[6][1], A_reg[1], B_reg[1], (2));
  C_reg[7][1] = vfmaq_laneq_f32(C_reg[7][1], A_reg[1], B_reg[1], (3));
  C_reg[8][0] = vfmaq_laneq_f32(C_reg[8][0], A_reg[0], B_reg[2], (0));
  C_reg[9][0] = vfmaq_laneq_f32(C_reg[9][0], A_reg[0], B_reg[2], (1));
  C_reg[10][0] = vfmaq_laneq_f32(C_reg[10][0], A_reg[0], B_reg[2], (2));
  C_reg[11][0] = vfmaq_laneq_f32(C_reg[11][0], A_reg[0], B_reg[2], (3));
  C_reg[8][1] = vfmaq_laneq_f32(C_reg[8][1], A_reg[1], B_reg[2], (0));
  C_reg[9][1] = vfmaq_laneq_f32(C_reg[9][1], A_reg[1], B_reg[2], (1));
  C_reg[10][1] = vfmaq_laneq_f32(C_reg[10][1], A_reg[1], B_reg[2], (2));
  C_reg[11][1] = vfmaq_laneq_f32(C_reg[11][1], A_reg[1], B_reg[2], (3));
}
vst1q_f32(&C.data[0], C_reg[0][0]);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg[0][1]);
vst1q_f32(&C.data[C.strides[0]], C_reg[1][0]);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg[1][1]);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg[2][0]);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[2][1]);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg[3][0]);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[3][1]);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg[4][0]);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[4][1]);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg[5][0]);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[5][1]);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg[6][0]);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[6][1]);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg[7][0]);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[7][1]);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg[8][0]);
vst1q_f32(&C.data[(8) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[8][1]);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg[9][0]);
vst1q_f32(&C.data[(9) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[9][1]);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg[10][0]);
vst1q_f32(&C.data[(10) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[10][1]);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg[11][0]);
vst1q_f32(&C.data[(11) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[11][1]);
}

// uk_8x4_a1True_b1True(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void uk_8x4_a1True_b1True( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg[4][2];
C_reg[0][0] = vld1q_f32(&C.data[0]);
C_reg[0][1] = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg[1][0] = vld1q_f32(&C.data[C.strides[0]]);
C_reg[1][1] = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg[2][0] = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg[2][1] = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[3][0] = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg[3][1] = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
float32x4_t A_reg[2];
float32x4_t B_reg[1];
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg[0] = vld1q_f32(&A[(k) * 8]);
  A_reg[1] = vld1q_f32(&A[(k) * 8 + 4]);
  B_reg[0] = vld1q_f32(&B[(k) * 4]);
  C_reg[0][0] = vfmaq_laneq_f32(C_reg[0][0], A_reg[0], B_reg[0], (0));
  C_reg[1][0] = vfmaq_laneq_f32(C_reg[1][0], A_reg[0], B_reg[0], (1));
  C_reg[2][0] = vfmaq_laneq_f32(C_reg[2][0], A_reg[0], B_reg[0], (2));
  C_reg[3][0] = vfmaq_laneq_f32(C_reg[3][0], A_reg[0], B_reg[0], (3));
  C_reg[0][1] = vfmaq_laneq_f32(C_reg[0][1], A_reg[1], B_reg[0], (0));
  C_reg[1][1] = vfmaq_laneq_f32(C_reg[1][1], A_reg[1], B_reg[0], (1));
  C_reg[2][1] = vfmaq_laneq_f32(C_reg[2][1], A_reg[1], B_reg[0], (2));
  C_reg[3][1] = vfmaq_laneq_f32(C_reg[3][1], A_reg[1], B_reg[0], (3));
}
vst1q_f32(&C.data[0], C_reg[0][0]);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg[0][1]);
vst1q_f32(&C.data[C.strides[0]], C_reg[1][0]);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg[1][1]);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg[2][0]);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[2][1]);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg[3][0]);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[3][1]);
}

// uk_8x8_a1True_b1True(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void uk_8x8_a1True_b1True( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg[8][2];
C_reg[0][0] = vld1q_f32(&C.data[0]);
C_reg[0][1] = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg[1][0] = vld1q_f32(&C.data[C.strides[0]]);
C_reg[1][1] = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg[2][0] = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg[2][1] = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[3][0] = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg[3][1] = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[4][0] = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg[4][1] = vld1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[5][0] = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg[5][1] = vld1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[6][0] = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg[6][1] = vld1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg[7][0] = vld1q_f32(&C.data[(7) * (C.strides[0])]);
C_reg[7][1] = vld1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])]);
float32x4_t A_reg[2];
float32x4_t B_reg[2];
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg[0] = vld1q_f32(&A[(k) * 8]);
  A_reg[1] = vld1q_f32(&A[(k) * 8 + 4]);
  B_reg[0] = vld1q_f32(&B[(k) * 8]);
  B_reg[1] = vld1q_f32(&B[(k) * 8 + 4]);
  C_reg[0][0] = vfmaq_laneq_f32(C_reg[0][0], A_reg[0], B_reg[0], (0));
  C_reg[1][0] = vfmaq_laneq_f32(C_reg[1][0], A_reg[0], B_reg[0], (1));
  C_reg[2][0] = vfmaq_laneq_f32(C_reg[2][0], A_reg[0], B_reg[0], (2));
  C_reg[3][0] = vfmaq_laneq_f32(C_reg[3][0], A_reg[0], B_reg[0], (3));
  C_reg[0][1] = vfmaq_laneq_f32(C_reg[0][1], A_reg[1], B_reg[0], (0));
  C_reg[1][1] = vfmaq_laneq_f32(C_reg[1][1], A_reg[1], B_reg[0], (1));
  C_reg[2][1] = vfmaq_laneq_f32(C_reg[2][1], A_reg[1], B_reg[0], (2));
  C_reg[3][1] = vfmaq_laneq_f32(C_reg[3][1], A_reg[1], B_reg[0], (3));
  C_reg[4][0] = vfmaq_laneq_f32(C_reg[4][0], A_reg[0], B_reg[1], (0));
  C_reg[5][0] = vfmaq_laneq_f32(C_reg[5][0], A_reg[0], B_reg[1], (1));
  C_reg[6][0] = vfmaq_laneq_f32(C_reg[6][0], A_reg[0], B_reg[1], (2));
  C_reg[7][0] = vfmaq_laneq_f32(C_reg[7][0], A_reg[0], B_reg[1], (3));
  C_reg[4][1] = vfmaq_laneq_f32(C_reg[4][1], A_reg[1], B_reg[1], (0));
  C_reg[5][1] = vfmaq_laneq_f32(C_reg[5][1], A_reg[1], B_reg[1], (1));
  C_reg[6][1] = vfmaq_laneq_f32(C_reg[6][1], A_reg[1], B_reg[1], (2));
  C_reg[7][1] = vfmaq_laneq_f32(C_reg[7][1], A_reg[1], B_reg[1], (3));
}
vst1q_f32(&C.data[0], C_reg[0][0]);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg[0][1]);
vst1q_f32(&C.data[C.strides[0]], C_reg[1][0]);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg[1][1]);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg[2][0]);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[2][1]);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg[3][0]);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[3][1]);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg[4][0]);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[4][1]);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg[5][0]);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[5][1]);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg[6][0]);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[6][1]);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg[7][0]);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])], C_reg[7][1]);
}

// example_sgemm_a1True_b1False(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[12, 8] @DRAM
// )
inline void uk_1xX_a1True_b1True( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float *C) {
for (int k = 0; k < KC; k++) {
  for (int j = 0; j < 12; j++) {
    for (int i = 0; i < 8; i++) {
      C[(j) * (8) + (i) * (1)] += A[(k) * (8) + (i) * (1)] * B[(k) * (12) + (j) * (1)];
    }
  }
}
}
