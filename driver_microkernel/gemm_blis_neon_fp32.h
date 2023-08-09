void gemm_microkernel_Cresident_neon_4x4_prefetch_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_neon_4x4_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float      *C, int ldC );

void gemm_microkernel_Cresident_neon_4x4_prefetch_unroll_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_neon_8x8_prefetch_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_neon_8x8_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC );
void gemm_microkernel_Cresident_neon_8x12_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC );
void gemm_microkernel_ABresident_neon_4x4_fp32( char, char, int, int, int, float, float *, int, float *, float, float * );

void fvtrans_float32_4x4( float32x4_t *, float32x4_t *, float32x4_t *, float32x4_t * );
void fvtrans_float32_8x8( float32x4_t *, float32x4_t *, float32x4_t *, float32x4_t *,
                          float32x4_t *, float32x4_t *, float32x4_t *, float32x4_t *,
                          float32x4_t *, float32x4_t *, float32x4_t *, float32x4_t *,
                          float32x4_t *, float32x4_t *, float32x4_t *, float32x4_t * );



float32_t dot(float32x4_t a, float32x4_t b);

