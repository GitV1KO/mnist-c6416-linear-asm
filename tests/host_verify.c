/* Host-only verification; this file never claims to execute DSP assembly. */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnn.h"
#include "cnn_kernels.h"
#include "cnn_data.h"

extern long legacy_dataset[500][28][28];
extern long legacy_labels[500];
extern long legacy_input[28][28];
extern long legacy_layer1_conv[24][24][5];
extern long legacy_layer1_pool[12][12][5];
extern long legacy_layer2_conv[8][8][10];
extern long legacy_layer2_pool[4][4][10];
extern long legacy_layer3_pool[10];
extern long legacy_layer3_conv[10];
extern int legacy_output;
extern void legacy_model_init(void);
extern void legacy_model_run(void);

static CnnState actual_state;
static unsigned int stub_calls = 0;

#if USE_LINEAR_ASM
/* Host C stub: confirms the external-symbol route; never executes .sa. */
int cnn_dotprod128(const short *x, const short *w)
{
    if (((uintptr_t)x & 31U) != 0 || ((uintptr_t)w & 31U) != 16U) {
        fprintf(stderr, "FAIL: kernel pointer alignment/bank-phase contract\n");
        exit(1);
    }
    ++stub_calls;
    return cnn_dotprod128_c(x, w);
}
#endif

static void fail(const char *what, int sample, int index)
{
    fprintf(stderr, "FAIL: %s sample=%d index=%d\n", what, sample, index);
    exit(1);
}

static void compare_legacy_layer(const int *actual, const long *expected,
                                 int count, const char *name, int sample)
{
    int i;
    for (i = 0; i < count; ++i) {
        if (expected[i] < INT_MIN || expected[i] > INT_MAX)
            fail("original intermediate does not fit int32", sample, i);
        if (actual[i] != expected[i]) {
            fprintf(stderr, "%s: new=%d original=%ld\n", name, actual[i], expected[i]);
            fail(name, sample, i);
        }
    }
}

static void check_image(const unsigned char image[784], int sample,
                        int compare_original)
{
    int j;
    unsigned int before = stub_calls;
    /* Poison the output to check every required field is written explicitly. */
    memset(&actual_state, 0xa5, sizeof(actual_state));
    if (cnn_forward(image, &actual_state) != 0) {
        fprintf(stderr, "%s\n", cnn_last_error());
        fail("production forward", sample, -1);
    }
    if (stub_calls - before != (USE_LINEAR_ASM ? 640U : 0U))
        fail("incorrect compile-time external kernel call routing", sample, -1);
    if (!compare_original) return;
    for (j = 0; j < 784; ++j) legacy_input[j / 28][j % 28] = image[j];
    /* Reset argmax for each image. The downloaded program carried this value
       across images, which can change the selected class only on exact ties. */
    legacy_output = 0;
    legacy_model_run();
    compare_legacy_layer(&actual_state.conv1[0][0][0],
                         &legacy_layer1_conv[0][0][0], 24 * 24 * 5, "conv1", sample);
    compare_legacy_layer(&actual_state.pool1[0][0][0],
                         &legacy_layer1_pool[0][0][0], 12 * 12 * 5, "pool1", sample);
    compare_legacy_layer(&actual_state.conv2[0][0][0],
                         &legacy_layer2_conv[0][0][0], 8 * 8 * 10, "conv2", sample);
    compare_legacy_layer(&actual_state.pool2[0][0][0],
                         &legacy_layer2_pool[0][0][0], 4 * 4 * 10, "pool2", sample);
    compare_legacy_layer(actual_state.avg, legacy_layer3_pool, 10, "avg", sample);
    for (j = 0; j < 10; ++j)
        if (actual_state.scores[j] != legacy_layer3_conv[j])
            fail("scores vs original", sample, j);
    if (actual_state.prediction != legacy_output)
        fail("prediction vs original", sample, -1);
}

static unsigned int random_state = 0x6416U;
static unsigned int next_random(void)
{
    random_state = random_state * 1664525U + 1013904223U;
    return random_state;
}

static void check_dot_product(void)
{
    short x[128] __attribute__((aligned(32)));
    short w_storage[144] __attribute__((aligned(32)));
    short *w = w_storage + 8;
    long long expected, absolute_sum, product;
    int trial, j;
    for (trial = 0; trial < 1004; ++trial) {
        expected = absolute_sum = 0;
        for (j = 0; j < 128; ++j) {
            if (trial == 0) { x[j] = 0; w[j] = -2672; }
            else if (trial == 1) { x[j] = 32767; w[j] = 511; }
            else if (trial == 2) { x[j] = -32768; w[j] = -511; }
            else if (trial == 3) { x[j] = 200; w[j] = -32768; }
            else {
                x[j] = (short)((int)(next_random() % 24001U) - 12000);
                w[j] = (short)((int)(next_random() % 1025U) - 512);
                if (j >= 125) { x[j] = 0; w[j] = 0; }
            }
            product = (long long)x[j] * w[j];
            expected += product;
            absolute_sum += product < 0 ? -product : product;
        }
        if (absolute_sum > INT_MAX || expected < INT_MIN || expected > INT_MAX)
            fail("test vector exceeds dot-product contract", trial, -1);
        if (cnn_dotprod128_c(x, w) != expected)
            fail("C dot-product", trial, -1);
    }
}

int main(void)
{
    unsigned char image[784];
    static const int expected_predictions[20] =
        {7,2,1,0,4,1,4,9,9,9,0,6,9,0,1,5,9,7,3,4};
    int sample, j, correct = 0;
    if (sizeof(int) != 4 || sizeof(short) != 2 || sizeof(long long) != 8)
        fail("unexpected host integer widths", -1, -1);
    if (CNN_SAMPLE_COUNT != 500) fail("expected temporary 500-image header", -1, -1);
    if (cnn_init() != 0) {
        fprintf(stderr, "%s\n", cnn_last_error());
        fail("cnn_init", -1, -1);
    }
    /* Exposed packed weights must preserve HWCN channel ordering and padding. */
    for (sample = 0; sample < 10; ++sample) {
        const short *weights = cnn_kernel_weights(sample);
        if (weights == NULL) fail("missing packed weights", sample, -1);
        if (((uintptr_t)weights & 31U) != 16U)
            fail("packed weight alignment/bank phase", sample, -1);
        for (j = 0; j < 128; ++j)
            if (weights[j] != (j < 125 ? cnn_w2[j * 10 + sample] : 0))
                fail("packed weights/channel order/padding", sample, j);
    }
    legacy_model_init();
    for (sample = 0; sample < CNN_SAMPLE_COUNT; ++sample) {
        for (j = 0; j < 784; ++j)
            if (cnn_images[sample][j] != legacy_dataset[sample][j / 28][j % 28])
                fail("embedded data vs original text", sample, j);
        if (cnn_labels[sample] != legacy_labels[sample])
            fail("embedded label vs original text", sample, -1);
        check_image(cnn_images[sample], sample, 1);
        if (sample < 20 && actual_state.prediction != expected_predictions[sample])
            fail("preserved first-20 prediction", sample, -1);
        if (actual_state.prediction == cnn_labels[sample]) ++correct;
    }
    /* Exercise blank, fully-on, alternating and deterministic random binary
       inputs, covering bounds beyond the particular MNIST test subset. */
    for (sample = 0; sample < 67; ++sample) {
        for (j = 0; j < 784; ++j) {
            if (sample == 0) image[j] = 0;
            else if (sample == 1) image[j] = 1;
            else if (sample == 2) image[j] = (unsigned char)(j & 1);
            else image[j] = (unsigned char)((next_random() >> 16) & 1U);
        }
        check_image(image, 500 + sample, 1);
    }
    memset(image, 0, sizeof(image));
    image[783] = 2;
    if (cnn_forward(image, &actual_state) == 0)
        fail("non-binary input was accepted", -1, -1);
    if (cnn_forward(NULL, &actual_state) == 0 ||
        cnn_forward(cnn_images[0], NULL) == 0)
        fail("invalid forward arguments were accepted", -1, -1);
    check_dot_product();
    if (correct != 485) fail("preserved 500-image accuracy", -1, -1);
    printf("PASS: USE_LINEAR_ASM=%d (%s): 500 MNIST + 67 binary edge/random images; all layers/scores match independent original C.\n",
           USE_LINEAR_ASM, USE_LINEAR_ASM ? "HOST C STUB, not .sa execution" : "host C");
    printf("PASS: 1004 C dot-product cases; original first-20 predictions preserved; external stub calls=%u.\n", stub_calls);
    printf("ACCURACY: %d/500 (%.2f%%). This is a host mathematical result, not DSP timing or board validation.\n", correct, correct / 5.0);
    return 0;
}
