#include "cnn_kernels.h"
#include "cnn_demo.h"

volatile int g_clock_result;
volatile int g_clock_expected;
volatile int g_clock_done;
volatile int g_clock_warmup;
volatile unsigned int g_clock_x_mod32, g_clock_w_mod32;

/* Separate functions, same optimization level as both kernels. No LTO/-pm.
 * The warm-up call is outside the measured region. */
void cnn_clock_warmup(const short *x, const short *w)
{
#if USE_LINEAR_ASM
    g_clock_warmup = cnn_dotprod128(x, w);
#else
    g_clock_warmup = cnn_dotprod128_c(x, w);
#endif
}

/* CCS Clock: only the selected direct call and its result store are measured.
 * In optimized code verify actual breakpoint addresses in Disassembly.
 * Do not single-step, log, verify data, or leave an inner breakpoint active
 * between CLOCK_START and CLOCK_STOP. */
void cnn_clock_demo(const short *x, const short *w)
{
#if USE_LINEAR_ASM
    g_clock_result = cnn_dotprod128(x, w);   /* CLOCK_START_ASM */
#else
    g_clock_result = cnn_dotprod128_c(x, w); /* CLOCK_START_C */
#endif
    g_clock_done = 1;                       /* CLOCK_STOP */
}
