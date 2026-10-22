#ifndef CNN_H
#define CNN_H
#include "cnn_config.h"

/* int is 32 bits on C6000 and the host; long is ABI dependent. */
typedef int CnnI32;
typedef long long CnnI64;
typedef char CnnShortMustBe16[(sizeof(short) == 2) ? 1 : -1];
typedef char CnnIntMustBe32[(sizeof(int) == 4) ? 1 : -1];
typedef char CnnLongLongMustBe64[(sizeof(long long) == 8) ? 1 : -1];

typedef struct CnnState {
    CnnI32 conv1[24][24][5];
    CnnI32 pool1[12][12][5];
    CnnI32 conv2[8][8][10];
    CnnI32 pool2[4][4][10];
    CnnI32 avg[10];
    CnnI64 scores[10];
    int prediction;
} CnnState;

/* 0 = success; -1 = invalid input, numeric contract violation. */
int cnn_init(void);
int cnn_forward(const unsigned char image[784], CnnState *state);
const char *cnn_last_error(void);
const short *cnn_kernel_weights(int channel);
/* Prepare one real first-image convolution patch, outside the Clock region. */
int cnn_prepare_demo(const CnnState *state);
const short *cnn_demo_input(void);

#endif
