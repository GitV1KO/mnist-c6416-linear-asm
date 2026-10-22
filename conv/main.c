#include "cnn.h"
#include "cnn_kernels.h"
#include "cnn_demo.h"
#include "cnn_golden.h"
#if CNN_USE_STDIO
#include <stdio.h>
#endif

#if CNN_SAMPLE_COUNT != CNN_GOLDEN_COUNT || CNN_SAMPLE_OFFSET != CNN_GOLDEN_OFFSET
#error Regenerate cnn_golden.h after changing the embedded dataset
#endif

volatile CnnReport g_cnn_report;
volatile int g_cnn_done;
CnnState g_cnn_state;

/* Preserve the explicit initialization verified on the real DSK6416.
 * Do not assume zero-initialized .bss when restarting from CCS Go Main. */
static void clear_cnn_report(void)
{
    volatile unsigned char *p =
        (volatile unsigned char *)&g_cnn_report;
    unsigned int i;
    for (i = 0; i < sizeof(g_cnn_report); ++i) {
        p[i] = 0;
    }
}

static void clear_cnn_state(void)
{
    volatile unsigned char *p = (volatile unsigned char *)&g_cnn_state;
    unsigned int i;
    for (i = 0; i < sizeof(g_cnn_state); ++i) p[i] = 0;
}

int main(void)
{
    int n, c, mismatch;
    const short *x;
    const short *w;

    clear_cnn_report();
    clear_cnn_state();
    g_cnn_done = 0;
    g_clock_result = 0;
    g_clock_expected = 0;
    g_clock_done = 0;
    g_clock_warmup = 0;
    g_clock_x_mod32 = 0;
    g_clock_w_mod32 = 0;
    g_cnn_report.assembly_enabled = USE_LINEAR_ASM;

    if (cnn_init() != 0) {
        ++g_cnn_report.failures;
    } else {
        /* Each image is inferred once, with the selected implementation. */
        for (n = 0; n < CNN_SAMPLE_COUNT; ++n) {
            if (cnn_forward(cnn_images[n], &g_cnn_state) != 0) {
                ++g_cnn_report.failures;
                break;
            }
            if (n == 0 && cnn_prepare_demo(&g_cnn_state) != 0) {
                ++g_cnn_report.failures;
                break;
            }
            /* Check all ten actual scores; these constants never feed CNN
             * computation. No second full inference is needed. */
            mismatch = g_cnn_state.prediction != cnn_golden_predictions[n];
            for (c = 0; c < 10; ++c)
                if (g_cnn_state.scores[c] != cnn_golden_scores[n][c]) mismatch = 1;
            if (mismatch) ++g_cnn_report.mismatches;
            g_cnn_report.predictions[n] = g_cnn_state.prediction;
            g_cnn_report.last_prediction = g_cnn_state.prediction;
            g_cnn_report.last_label = cnn_labels[n];
            if (g_cnn_state.prediction == cnn_labels[n]) ++g_cnn_report.correct;
            ++g_cnn_report.samples;
#if CNN_USE_STDIO
            printf("sample=%d predict=%d label=%d\n", n,
                   g_cnn_state.prediction, (int)cnn_labels[n]);
#endif
        }
    }

    if (!g_cnn_report.failures && !g_cnn_report.mismatches
        && g_cnn_report.samples == CNN_SAMPLE_COUNT) {
        /* Same real image-0 patch and channel-0 weights in both builds.
         * Captured before later images could overwrite the first state. */
        x = cnn_demo_input();
        w = cnn_kernel_weights(0);
        if (!x || !w) {
            ++g_cnn_report.failures;
        } else {
            g_clock_x_mod32 = (unsigned int)((unsigned long)x & 31UL);
            g_clock_w_mod32 = (unsigned int)((unsigned long)w & 31UL);
            if (g_clock_x_mod32 != 0u || g_clock_w_mod32 != 16u) {
                ++g_cnn_report.failures;
            } else {
                cnn_clock_warmup(x, w); /* One selected-kernel warm-up, untimed. */
                cnn_clock_demo(x, w);   /* Set the two breakpoints INSIDE here. */
                /* Correctness check is after CLOCK_STOP, never timed. */
                g_clock_expected = cnn_dotprod128_c(x, w);
                if (!g_clock_done || g_clock_result != g_clock_expected)
                    ++g_cnn_report.mismatches;
            }
        }
    }

    g_cnn_done = g_cnn_report.failures || g_cnn_report.mismatches ? -1 : 1;
#if CNN_USE_STDIO
    printf("samples=%u correct=%u mismatches=%u failures=%u assembly=%u\n",
           g_cnn_report.samples, g_cnn_report.correct, g_cnn_report.mismatches,
           g_cnn_report.failures, g_cnn_report.assembly_enabled);
    printf("clock_result=%d expected=%d done=%d cnn_done=%d\n",
           g_clock_result, g_clock_expected, g_clock_done, g_cnn_done);
    if (g_cnn_report.failures) printf("error: %s\n", cnn_last_error());
#endif
    return g_cnn_done == 1 ? 0 : 1;
}
