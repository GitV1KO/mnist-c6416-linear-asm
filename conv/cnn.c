#include <limits.h>
#include <string.h>
#include "cnn.h"
#include "cnn_kernels.h"
/* Angle brackets allow tests to supply a different generated data header. */
#include <cnn_data.h>

#if CNN_TI_TARGET
#pragma DATA_ALIGN(packed_weights_storage, 32)
#pragma DATA_ALIGN(patch, 32)
#pragma DATA_ALIGN(demo_patch, 32)
#endif
#if defined(__GNUC__) || defined(__clang__)
#define CNN_ALIGN32 __attribute__((aligned(32)))
#else
#define CNN_ALIGN32
#endif
/* Each row is 288 bytes (a multiple of 32). Its active 128-short
 * weights begin 16 bytes into the row, on a different L1D bank phase
 * from patch. The same pointers are passed to C and assembly. */
static short packed_weights_storage[10][CNN_WEIGHT_ROW_SHORTS] CNN_ALIGN32;
static short patch[CNN_DOT_LENGTH] CNN_ALIGN32;
static short demo_patch[CNN_DOT_LENGTH] CNN_ALIGN32;
static int demo_ready;
static int initialized = 0;
static const char *last_error = "not initialized";

static int fail(const char *message)
{
    last_error = message;
    return -1;
}
const char *cnn_last_error(void) { return last_error; }

int cnn_init(void)
{
    int n, k;
    CnnI64 upper, abs_sum, b;
    initialized = 0;
    demo_ready = 0;
    last_error = "OK";
    /* Prove that binary input -> conv1/ReLU fits signed 16 bits. */
    for (n = 0; n < 5; ++n) {
        upper = cnn_b1[n];
        for (k = 0; k < 25; ++k)
            if (cnn_w1[k * 5 + n] > 0) upper += cnn_w1[k * 5 + n];
        if (upper > SHRT_MAX)
            return fail("conv1 bound exceeds signed short; requantize weights");
    }
    memset(packed_weights_storage, 0, sizeof(packed_weights_storage));
    memset(patch, 0, sizeof(patch));
    memset(demo_patch, 0, sizeof(demo_patch));
    for (n = 0; n < 10; ++n) {
        abs_sum = 0;
        for (k = 0; k < CNN_DOT_VALID; ++k) {
            packed_weights_storage[n][CNN_WEIGHT_OFFSET_SHORTS + k] = cnn_w2[k * 10 + n];
            b = cnn_w2[k * 10 + n];
            abs_sum += b < 0 ? -b : b;
        }
        b = cnn_b2[n];
        if (b < 0) b = -b;
        /* Stronger than the binary-input bound: valid for any short patch.
         * Includes bias, ensures every DOTP2 and partial sum is safe. */
        if (abs_sum * 32768LL + b > INT_MAX)
            return fail("conv2 32-bit accumulation bound exceeded");
    }
    if (((unsigned long)patch & 31UL) != 0UL
        || ((unsigned long)demo_patch & 31UL) != 0UL
        || ((unsigned long)(packed_weights_storage[0] + CNN_WEIGHT_OFFSET_SHORTS)
            & 31UL) != 16UL)
        return fail("kernel alignment / bank phase contract violated");
    initialized = 1;
    last_error = "OK";
    return 0;
}

const short *cnn_kernel_weights(int channel)
{
    if (!initialized || channel < 0 || channel >= 10) return 0;
    return packed_weights_storage[channel] + CNN_WEIGHT_OFFSET_SHORTS;
}

static void conv1_run(const unsigned char *image, CnnState *s)
{
    int n, h, w, x, y, sum;
    for (n = 0; n < 5; ++n)
        for (h = 0; h < 24; ++h)
            for (w = 0; w < 24; ++w) {
                sum = 0;
                for (x = 0; x < 5; ++x)
                    for (y = 0; y < 5; ++y)
                        sum += (int)image[(h + x) * 28 + w + y]
                             * (int)cnn_w1[(x * 5 + y) * 5 + n];
                sum += cnn_b1[n];
                s->conv1[h][w][n] = sum > 0 ? sum : 0;
            }
}

static void pool1_run(CnnState *s)
{
    int h, w, c, x, y, maximum;
    for (h = 0; h < 12; ++h)
        for (w = 0; w < 12; ++w)
            for (c = 0; c < 5; ++c) {
                maximum = 0;
                for (x = 0; x < 2; ++x)
                    for (y = 0; y < 2; ++y)
                        if (s->conv1[2*h+x][2*w+y][c] > maximum)
                            maximum = s->conv1[2*h+x][2*w+y][c];
                s->pool1[h][w][c] = maximum;
            }
}

static int pack_window(const CnnState *s, int h, int w, short *destination)
{
    int x, y, c, k = 0, value;
    for (x = 0; x < 5; ++x)
        for (y = 0; y < 5; ++y)
            for (c = 0; c < 5; ++c) {
                value = s->pool1[h+x][w+y][c];
                if (value < SHRT_MIN || value > SHRT_MAX)
                    return fail("patch value cannot be represented as short");
                destination[k++] = (short)value;
            }
    destination[125] = destination[126] = destination[127] = 0;
    return 0;
}

int cnn_prepare_demo(const CnnState *state)
{
    if (!initialized || !state) return fail("invalid demo state");
    if (demo_ready) return fail("demo input already captured; call cnn_init to reset");
    if (pack_window(state, 0, 0, demo_patch) != 0) return -1;
    demo_ready = 1;
    return 0;
}

const short *cnn_demo_input(void)
{
    return initialized && demo_ready ? demo_patch : 0;
}

static int conv2_run(CnnState *s)
{
    int h, w, n, sum;
    const short *weights;
    for (h = 0; h < 8; ++h)
        for (w = 0; w < 8; ++w) {
            if (pack_window(s, h, w, patch) != 0) return -1;
            for (n = 0; n < 10; ++n) {
                weights = packed_weights_storage[n] + CNN_WEIGHT_OFFSET_SHORTS;
                /* Exactly one direct call is compiled; switch in cnn_config.h. */
#if USE_LINEAR_ASM
                sum = cnn_dotprod128(patch, weights);
#else
                sum = cnn_dotprod128_c(patch, weights);
#endif
                sum += cnn_b2[n];
                s->conv2[h][w][n] = sum > 0 ? sum : 0;
            }
        }
    return 0;
}

static void pool2_run(CnnState *s)
{
    int h, w, c, x, y, maximum;
    for (h = 0; h < 4; ++h)
        for (w = 0; w < 4; ++w)
            for (c = 0; c < 10; ++c) {
                maximum = 0;
                for (x = 0; x < 2; ++x)
                    for (y = 0; y < 2; ++y)
                        if (s->conv2[2*h+x][2*w+y][c] > maximum)
                            maximum = s->conv2[2*h+x][2*w+y][c];
                s->pool2[h][w][c] = maximum;
            }
}

static int head_run(CnnState *s)
{
    int c, h, w, n;
    CnnI64 sum, product;
    for (c = 0; c < 10; ++c) {
        s->avg[c] = 0;
        /* Preserve upstream's divide-each-element integer rounding. */
        for (h = 0; h < 4; ++h)
            for (w = 0; w < 4; ++w)
                s->avg[c] += s->pool2[h][w][c] / 16;
    }
    for (n = 0; n < 10; ++n) {
        sum = 0;
        for (c = 0; c < 10; ++c) {
            product = (CnnI64)s->avg[c] * cnn_w3[c*10+n];
            if (product < -549755813888LL || product > 549755813887LL)
                return fail("output product exceeds upstream COFF long range");
            sum += product;
            /* Reject inputs that would exceed the original COFF long range. */
            if (sum < -549755813888LL || sum > 549755813887LL)
                return fail("output accumulation exceeds upstream COFF long range");
        }
        sum += cnn_b3[n];
        if (sum < -549755813888LL || sum > 549755813887LL)
            return fail("output bias exceeds upstream COFF long range");
        s->scores[n] = sum;
    }
    s->prediction = 0;
    for (n = 1; n < 10; ++n)
        if (s->scores[n] > s->scores[s->prediction]) s->prediction = n;
    return 0;
}

int cnn_forward(const unsigned char image[784], CnnState *state)
{
    int i;
    if (!initialized) return fail("call cnn_init first");
    if (!image || !state) return fail("null image or state");
    for (i = 0; i < 784; ++i)
        if (image[i] > 1) return fail("input must be binary 0/1");
    conv1_run(image, state);
    pool1_run(state);
    if (conv2_run(state) != 0) return -1;
    pool2_run(state);
    if (head_run(state) != 0) return -1;
    last_error = "OK";
    return 0;
}

