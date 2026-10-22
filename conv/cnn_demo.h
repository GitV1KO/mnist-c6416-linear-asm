#ifndef CNN_DEMO_H
#define CNN_DEMO_H
#include "cnn_config.h"
#include <cnn_data.h>

/* No software timer statistics. All fields are explicitly cleared in main. */
typedef struct CnnReport {
    unsigned int samples, correct, mismatches, failures;
    unsigned int assembly_enabled;
    int predictions[CNN_SAMPLE_COUNT];
    int last_prediction, last_label;
} CnnReport;

extern volatile CnnReport g_cnn_report;
extern volatile int g_cnn_done;
extern volatile int g_clock_result;
extern volatile int g_clock_expected;
extern volatile int g_clock_done;
extern volatile int g_clock_warmup;
extern volatile unsigned int g_clock_x_mod32, g_clock_w_mod32;

void cnn_clock_warmup(const short *x, const short *w);
void cnn_clock_demo(const short *x, const short *w);

#endif
