#!/usr/bin/env python3
"""Compile both firmware branches on the HOST and check reset/call routing.

USE_LINEAR_ASM=1 links an explicitly marked HOST C stub, never the .sa source.
Both builds include the real main.c and clock_demo.c; no TI tool is invoked.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HARNESS = r'''
#define main cnn_production_main
#include "main.c"
#undef main
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned int stub_calls;
static CnnState demo_check_state;

static void check(int condition, const char *message)
{
    if (!condition) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}

#if USE_LINEAR_ASM
/* Explicit HOST C stub; this is NOT the .sa implementation. */
int cnn_dotprod128(const short *x, const short *w)
{
    check(((uintptr_t)x & 31U) == 0 && ((uintptr_t)w & 31U) == 16,
          "external call violated x%32=0, w%32=16");
    ++stub_calls;
    return cnn_dotprod128_c(x, w);
}
#endif

int main(void)
{
    static const int predictions[20] =
        {7,2,1,0,4,1,4,9,9,9,0,6,9,0,1,5,9,7,3,4};
    short captured_demo[128];
    const short *demo;
    unsigned int before;
    int run, i, x, y, c, k;
    check(CNN_SAMPLE_COUNT == 20 && CNN_SAMPLE_OFFSET == 0,
          "this regression check expects the shipped first 20 images");
    for (run = 0; run < 2; ++run) {
        /* CCS Go Main can preserve globals. Deliberately dirty them first. */
        memset((void *)&g_cnn_report, 0xa5, sizeof(g_cnn_report));
        memset(&g_cnn_state, 0xa5, sizeof(g_cnn_state));
        g_cnn_done = g_clock_result = g_clock_expected = g_clock_done = -99;
        g_clock_warmup = -99;
        g_clock_x_mod32 = g_clock_w_mod32 = 99U;
        before = stub_calls;
        check(cnn_production_main() == 0, "production main returned failure");
        check(g_cnn_done == 1 && g_cnn_report.samples == 20 &&
              g_cnn_report.correct == 19 && g_cnn_report.mismatches == 0 &&
              g_cnn_report.failures == 0 &&
              g_cnn_report.assembly_enabled == USE_LINEAR_ASM,
              "main counters were not explicitly reset/reproduced");
        check(stub_calls - before == (USE_LINEAR_ASM ? 12802U : 0U),
              "main must run 20*640 kernel calls plus one warmup and one Clock demo");
        for (i = 0; i < 20; ++i)
            check(g_cnn_report.predictions[i] == predictions[i], "first-20 predictions changed");
        check(g_clock_done == 1 && g_clock_result == -3813844 &&
              g_clock_expected == g_clock_result && g_clock_warmup == g_clock_result &&
              g_clock_x_mod32 == 0U && g_clock_w_mod32 == 16U,
              "Clock demo result/reset/alignment mismatch");
        demo = cnn_demo_input();
        check(demo != NULL && ((uintptr_t)demo & 31U) == 0, "demo input missing/misaligned");
        memcpy(captured_demo, demo, sizeof(captured_demo));
        check(cnn_forward(cnn_images[0], &demo_check_state) == 0, "image-0 forward failed");
        k = 0;
        for (x = 0; x < 5; ++x)
            for (y = 0; y < 5; ++y)
                for (c = 0; c < 5; ++c)
                    check(captured_demo[k++] == demo_check_state.pool1[x][y][c],
                          "demo is not the first real image-0 window");
        check(captured_demo[125] == 0 && captured_demo[126] == 0 && captured_demo[127] == 0,
              "demo zero padding changed");
        check(cnn_forward(cnn_images[1], &demo_check_state) == 0, "image-1 forward failed");
        check(memcmp(captured_demo, cnn_demo_input(), sizeof(captured_demo)) == 0,
              "later inference overwrote the saved demo input");
        check(cnn_prepare_demo(&demo_check_state) != 0,
              "demo must reject recapture until cnn_init resets it");
    }
    printf("PASS: HOST %s main twice after poisoned globals: 19/20, all 200 golden scores match, failures=0.\n",
           USE_LINEAR_ASM ? "C-STUB assembly-call branch" : "C branch");
    printf("PASS: stable real-image demo patch, bank phases and Clock result=-3813844; selected calls are direct.\n");
    return 0;
}
'''


def verify_source_contracts(project: Path) -> None:
    main_source = (project / "conv" / "main.c").read_text()
    for name in ("g_cnn_done", "g_clock_result", "g_clock_expected", "g_clock_done",
                 "g_clock_warmup", "g_clock_x_mod32", "g_clock_w_mod32"):
        if not re.search(r"\b" + name + r"\s*=\s*0\s*;", main_source):
            raise ValueError(f"main lacks explicit clearing of {name}")
    for clear, target in (("clear_cnn_report", "g_cnn_report"), ("clear_cnn_state", "g_cnn_state")):
        if main_source.index(clear + "();") > main_source.index("if (cnn_init()"):
            raise ValueError(f"{target} is cleared too late")
        body = re.search(r"static void " + clear + r"\(void\)\s*\{(.*?)\n\}", main_source, re.S)
        if not body or "volatile unsigned char" not in body.group(1) or "p[i] = 0" not in body.group(1):
            raise ValueError(f"{target} lacks an explicit byte clear")
    linker = (project / "conv" / "C6416.cmd").read_text()
    for region, origin, length in (("VECS", 0, 0x400), ("IRAM", 0x400, 0x2FC00)):
        match = re.search(region + r"\s*:\s*org\s*=\s*(0x[0-9a-fA-F]+)\s*,\s*len\s*=\s*(0x[0-9a-fA-F]+)", linker)
        if not match or tuple(int(x, 16) for x in match.groups()) != (origin, length):
            raise ValueError(f"unexpected {region} memory range")
    if re.search(r":>\s*VECS\b", linker):
        raise ValueError("ordinary program section must not occupy the reserved VECS region")
    print("PASS: source explicitly clears report/state/Clock fields; IRAM starts at 0x400 and VECS [0,0x400) is reserved.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clang", default="clang")
    args = parser.parse_args()
    project = args.project.resolve()
    compiler = shutil.which(args.clang)
    if not compiler:
        raise RuntimeError("clang is not installed or is not on PATH")
    verify_source_contracts(project)
    with tempfile.TemporaryDirectory(prefix="cnn-host-firmware-") as name:
        temp = Path(name)
        harness = temp / "firmware_check.c"
        harness.write_text(HARNESS)
        for use_asm in (0, 1):
            executable = temp / f"firmware-{use_asm}"
            target_flags = ["-D_TMS320C6X=1"] if use_asm else []
            subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-Wno-unknown-pragmas", f"-DUSE_LINEAR_ASM={use_asm}",
                            "-DCNN_USE_STDIO=0", *target_flags, "-I", str(project / "conv"),
                            str(project / "conv" / "cnn.c"), str(project / "conv" / "cnn_dotprod128_c.c"),
                            str(project / "conv" / "clock_demo.c"), str(harness),
                            "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)
    print("LIMIT: Both programs ran on the HOST. The assembly branch linked a HOST C stub; no .sa, TI compiler, DSP instruction timing or hardware was executed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)
