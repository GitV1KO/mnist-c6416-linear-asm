#ifndef CNN_KERNELS_H
#define CNN_KERNELS_H
#include "cnn_config.h"

/* Both buffers: 128 signed shorts, read-only during each call.
 * Alignment / L1D bank phase: x address % 32 == 0; w address % 32 == 16.
 * Thus both are 8-byte aligned for C64x LDDW. The C kernel receives the
 * same alignment information as the assembly optimizer.
 * Required: sum(abs(x[i]*w[i])) <= 2147483647. Return: signed 32-bit sum.
 * CNN uses 125 valid elements plus 3 zero elements. */
int cnn_dotprod128_c(const short *x, const short *w);
#if USE_LINEAR_ASM
/* COFF link name: _cnn_dotprod128, implemented in cnn_dotprod128.sa. */
int cnn_dotprod128(const short *x, const short *w);
#endif

#endif
