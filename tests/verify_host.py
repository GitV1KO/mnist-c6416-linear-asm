#!/usr/bin/env python3
"""Compile and run mathematical checks on the host, never on the DSP.

Requires standard-library Python 3 and clang. A temporary 500-image header is
generated for verification; the shipped DSP header is left unchanged. The
downloaded C source is compiled with symbol renaming and file-path replacement
only, preserving its convolution/pooling arithmetic as an independent oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        diagnostic = result.stderr
        if len(diagnostic) > 20000:
            diagnostic = diagnostic[:5000] + "\n[compiler output shortened]\n" + diagnostic[-15000:]
        print(diagnostic, end="", file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command[0]}")


def values(path: Path) -> list[int]:
    return [int(word) for word in path.read_text().split()]


def verify_exported_header(header: Path, data_dir: Path) -> None:
    """Independently verify converter contents against the source text files."""
    text = header.read_text()
    count_match = re.search(r"#\s*define\s+CNN_SAMPLE_COUNT\s+(\d+)", text)
    if not count_match:
        raise ValueError(f"missing CNN_SAMPLE_COUNT in {header}")
    count = int(count_match.group(1))
    offset_match = re.search(r"#\s*define\s+CNN_SAMPLE_OFFSET\s+(\d+)", text)
    if not offset_match:
        raise ValueError(f"missing CNN_SAMPLE_OFFSET in {header}")
    offset = int(offset_match.group(1))
    shapes = {
        "cnn_w1": ("conv1_w.txt", 125), "cnn_b1": ("conv1_b.txt", 5),
        "cnn_w2": ("conv2_w.txt", 1250), "cnn_b2": ("conv2_b.txt", 10),
        "cnn_w3": ("conv3_w.txt", 100), "cnn_b3": ("conv3_b.txt", 10),
        "cnn_images": ("data.txt", count * 784),
        "cnn_labels": ("label.txt", count),
    }
    for name, (filename, length) in shapes.items():
        match = re.search(r"\b" + name + r"\s*(?:\[[^\]]*\]\s*)+\s*=\s*\{(.*?)\}\s*;", text, re.S)
        if not match:
            raise ValueError(f"missing initializer {name} in {header}")
        body = re.sub(r"/\*.*?\*/|//[^\n]*", "", match.group(1), flags=re.S)
        actual = [int(x) for x in re.findall(r"-?\d+", body)]
        start = offset * 784 if name == "cnn_images" else offset if name == "cnn_labels" else 0
        expected = values(data_dir / filename)[start:start + length]
        if actual != expected or len(actual) != length:
            raise ValueError(f"{name} does not match {filename} ({length} elements)")
    print(f"PASS: shipped header matches original weights and images/labels [{offset}, {offset + count}).")


def verify_golden_metadata(header: Path, golden: Path) -> None:
    data_text, golden_text = header.read_text(), golden.read_text()
    for data_name, golden_name in (("CNN_SAMPLE_COUNT", "CNN_GOLDEN_COUNT"),
                                    ("CNN_SAMPLE_OFFSET", "CNN_GOLDEN_OFFSET")):
        expected = re.search(r"#\s*define\s+" + data_name + r"\s+(\d+)", data_text)
        actual = re.search(r"#\s*define\s+" + golden_name + r"\s+(\d+)", golden_text)
        if not expected or not actual or expected.group(1) != actual.group(1):
            raise ValueError(f"golden {golden_name} is stale; regenerate golden data")
    recorded = re.search(r'#\s*define\s+CNN_GOLDEN_DATA_SHA256\s+"([0-9a-f]{64})"', golden_text)
    digest = hashlib.sha256(header.read_bytes()).hexdigest()
    if not recorded or recorded.group(1) != digest:
        raise ValueError("golden data-header SHA256 is stale; run tests/generate_golden.py")
    print("PASS: golden COUNT/OFFSET and complete data-header SHA256 match the shipped header.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--original-source", type=Path)
    parser.add_argument("--clang", default="clang")
    args = parser.parse_args()
    project = args.project.resolve()
    data_dir = (args.data_dir or project / "data").resolve()
    original = (args.original_source or project / "reference" / "main.c").resolve()
    compiler = shutil.which(args.clang)
    if not compiler:
        raise RuntimeError("clang is not installed or is not on PATH")
    exported_candidates = sorted((project / "conv").rglob("cnn_data.h"))
    if len(exported_candidates) != 1:
        raise RuntimeError("expected exactly one exported conv/**/cnn_data.h")
    verify_exported_header(exported_candidates[0], data_dir)
    verify_golden_metadata(exported_candidates[0], project / "conv" / "cnn_golden.h")
    names = [
        "dataset", "labels", "conv1_w", "conv1_b", "conv2_w", "conv2_b",
        "conv3_w", "conv3_b", "input", "layer1_conv", "layer1_pool",
        "layer2_conv", "layer2_pool", "layer3_pool", "layer3_conv", "output",
        "conv1", "pooling1", "conv2", "pooling2", "pooling3", "conv3",
        "model_run", "get_weights", "get_dataset", "get_labels", "model_init", "main",
    ]
    with tempfile.TemporaryDirectory(prefix="cnn-host-verify-") as name:
        temp = Path(name)
        include = temp / "include"
        include.mkdir()
        run([sys.executable, str(project / "tools" / "embed_data.py"),
             "--data-dir", str(data_dir), "--count", "500",
             "--output", str(include / "cnn_data.h")])
        source = original.read_text()
        source = source.replace(r"..\\data\\", str(data_dir) + "/")
        # C source contains the backslash characters escaped as '\\'.
        if r"..\\data\\" in source:
            raise RuntimeError("original-source Windows file paths were not replaced")
        renamed = "".join(f"#define {symbol} legacy_{symbol}\n" for symbol in names) + source
        legacy = temp / "legacy.c"
        legacy.write_text(renamed)
        env = os.environ.copy()
        # Apple's ASan runtime rejects LeakSanitizer; address/UB checks still run.
        env["ASAN_OPTIONS"] = ("detect_leaks=0" if sys.platform == "darwin" else "detect_leaks=1") + ":halt_on_error=1"
        env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        for use_asm in (0, 1):
            executable = temp / f"verify-{use_asm}"
            target_flags = ["-D_TMS320C6X=1"] if use_asm else []
            run([compiler, "-std=c99", "-O1", "-g", "-Wall", "-Wextra",
                 "-Wno-unknown-pragmas", "-fsanitize=address,undefined",
                 "-fno-sanitize-recover=all", "-fno-omit-frame-pointer",
                 f"-DUSE_LINEAR_ASM={use_asm}", "-DCNN_USE_TIMER0=0", *target_flags,
                 "-I", str(include), "-I", str(project / "conv"),
                 str(project / "conv" / "cnn.c"), str(project / "conv" / "cnn_dotprod128_c.c"),
                 str(project / "tests" / "host_verify.c"), str(legacy), "-o", str(executable)])
            run([str(executable)], env=env)
    print("PASS: clang AddressSanitizer and UndefinedBehaviorSanitizer reported no error.")
    print("LIMIT: This does not validate TI CCS compilation, C64x instruction scheduling, ABI, hardware timing, or execution on a DSP board.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
