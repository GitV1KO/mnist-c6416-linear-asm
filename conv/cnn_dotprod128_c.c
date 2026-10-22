#include "cnn_kernels.h"
#if CNN_TI_TARGET && !defined(__GNUC__) && !defined(__clang__)
#include <c6x.h>
#endif

/* Separate translation unit: same call boundary as the .sa baseline. */
int cnn_dotprod128_c(const short *x, const short *w)
{
    int i;
    int sum = 0;
#if CNN_TI_TARGET && !defined(__GNUC__) && !defined(__clang__)
    /* Compiler facts, not runtime work. Keep C fully optimized and give it
     * the same true alignment/bank information as .mptr in the .sa file. */
    _nassert(((unsigned int)x & 31u) == 0u);
    _nassert(((unsigned int)w & 31u) == 16u);
#endif
    for (i = 0; i < CNN_DOT_LENGTH; ++i)
        sum += (int)x[i] * (int)w[i];
    return sum;
}
