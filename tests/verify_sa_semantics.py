#!/usr/bin/env python3
"""Execute data semantics parsed from the actual C64x .sa source.

This small interpreter is deliberately limited to the instructions used by the
kernel. It checks signed packed-word loads, postincrement, DOTP2, accumulation,
loop bounds and the returned value against Python and compiled C. It is NOT a
TI assembler, instruction scheduler, ABI checker, cycle simulator, or DSP run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from dataclasses import dataclass
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile


INT_MIN = -(1 << 31)
INT_MAX = (1 << 31) - 1


def signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


@dataclass(frozen=True)
class Instruction:
    opcode: str
    args: tuple[str, ...]
    predicate: str | None
    line: int


class LinearAssembly:
    def __init__(self, path: Path):
        self.instructions: list[Instruction] = []
        self.labels: dict[str, int] = {}
        self.arguments: list[str] = []
        self.trip: int | None = None
        self.mptr: dict[str, tuple[str, int, int]] = {}
        found_export = False
        for number, raw in enumerate(path.read_text().splitlines(), 1):
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            match = re.match(r"^(\w+):\s*(.*)$", line)
            if match:
                self.labels[match.group(1)] = len(self.instructions)
                line = match.group(2).strip()
            if not line:
                continue
            if line.startswith(".global"):
                found_export |= "_cnn_dotprod128" in line.split()
                continue
            if line.startswith(".cproc"):
                self.arguments = [x.strip() for x in line[len(".cproc"):].split(",")]
                continue
            if line.startswith(".return"):
                self.instructions.append(Instruction("RETURN", (line[len(".return"):].strip(),), None, number))
                continue
            if line.startswith("."):
                directive = line.split()[0]
                if directive == ".trip":
                    trip = [int(x.strip(), 0) for x in line[len(directive):].split(",")]
                    if len(trip) != 2 or trip[0] != trip[1] or trip[0] not in (16, 32):
                        raise ValueError("expected a fixed 16- or 32-iteration optimized kernel")
                    self.trip = trip[0]
                if directive == ".mptr":
                    match = re.fullmatch(r"\.mptr\s+(\w+),\s*(\w+)\+(\d+),\s*(\d+)", line)
                    if not match:
                        raise ValueError(f"unsupported .mptr metadata: {line}")
                    register, base, phase, stride = match.groups()
                    self.mptr[register] = (base, int(phase), int(stride))
                if directive not in {".sect", ".reg", ".rega", ".regb", ".mptr", ".trip", ".endproc"}:
                    raise ValueError(f"unsupported .sa directive at line {number}: {line}")
                continue
            predicate = None
            match = re.match(r"^\[\s*(!?\w+)\s*\]\s*(.*)$", line)
            if match:
                predicate, line = match.groups()
            match = re.match(r"^(\w+)(?:\s+\.[A-Za-z0-9]+)?\s+(.*)$", line)
            if not match:
                raise ValueError(f"cannot parse instruction at line {number}: {line}")
            opcode, operands = match.groups()
            opcode = opcode.upper()
            if opcode not in {"MVK", "ZERO", "LDW", "LDDW", "DOTP2", "ADD", "SUB", "B"}:
                raise ValueError(f"unsupported instruction at line {number}: {opcode}")
            self.instructions.append(Instruction(opcode, tuple(x.strip() for x in operands.split(",")), predicate, number))
        if not found_export or len(self.arguments) != 2:
            raise ValueError("expected exported COFF _cnn_dotprod128 with two .cproc arguments")
        if not self.instructions or self.instructions[-1].opcode != "RETURN":
            raise ValueError("kernel must end with a .return")
        if self.trip is None:
            raise ValueError("missing fixed trip metadata")
        if self.mptr.get(self.arguments[0]) != ("bankbase", 0, 8) or self.mptr.get(self.arguments[1]) != ("bankbase", 16, 8):
            raise ValueError(".mptr metadata must match x%32=0, w%32=16, stride=8")

    def execute(self, x: list[int], w: list[int]) -> int:
        if len(x) != 128 or len(w) != 128:
            raise ValueError("test vectors must contain exactly 128 shorts")
        if any(a < -32768 or a > 32767 for a in x + w):
            raise ValueError("test element does not fit signed short")
        if sum(abs(a * b) for a, b in zip(x, w)) > INT_MAX:
            raise ValueError("test vector violates the absolute-accumulation contract")
        memory: dict[int, int] = {}
        for base, data in ((0x1000, x), (0x2010, w)):
            for i, item in enumerate(data):
                memory[base + 2 * i] = item & 0xFF
                memory[base + 2 * i + 1] = (item >> 8) & 0xFF
        reads: Counter[int] = Counter()
        registers = dict(zip(self.arguments, (0x1000, 0x2010)))
        counts = dict.fromkeys(("LDW", "LDDW", "DOTP2", "ADD", "SUB", "B"), 0)
        pc, steps = 0, 0

        def value(token: str) -> int:
            if token in registers:
                return registers[token]
            try:
                return int(token, 0)
            except ValueError as error:
                raise ValueError(f"undefined register/operand {token}") from error

        while pc < len(self.instructions):
            steps += 1
            if steps > 1000:
                raise ValueError("instruction limit exceeded: broken branch/loop")
            instruction = self.instructions[pc]
            pc += 1
            op, args = instruction.opcode, instruction.args
            if instruction.predicate:
                negate = instruction.predicate.startswith("!")
                predicate = instruction.predicate.lstrip("!")
                condition = value(predicate) != 0
                if condition == negate:
                    continue
            if op in counts:
                counts[op] += 1
            if op == "MVK":
                registers[args[1]] = signed(value(args[0]), 32)
            elif op == "ZERO":
                registers[args[0]] = 0
            elif op in {"LDW", "LDDW"}:
                match = re.fullmatch(r"\*(\w+)\+\+(?:\[(\d+)\])?", args[0])
                if not match:
                    raise ValueError(f"unsupported {op} addressing at line {instruction.line}: {args[0]}")
                pointer, increment = match.groups()
                address = value(pointer)
                width = 8 if op == "LDDW" else 4
                if address % width or any(address + i not in memory for i in range(width)):
                    raise ValueError(f"unaligned/out-of-bounds {op} at .sa line {instruction.line}")
                loaded = sum(memory[address + i] << (8 * i) for i in range(width))
                reads.update(range(address, address + width))
                if op == "LDDW":
                    pair = args[1].split(":")
                    if len(pair) != 2:
                        raise ValueError("LDDW must write a high:low register pair")
                    high, low = pair
                    registers[low] = signed(loaded, 32)
                    registers[high] = signed(loaded >> 32, 32)
                else:
                    registers[args[1]] = signed(loaded, 32)
                registers[pointer] += width * (int(increment) if increment else 1)
            elif op == "DOTP2":
                a, b = value(args[0]), value(args[1])
                product = signed(a, 16) * signed(b, 16) + signed(a >> 16, 16) * signed(b >> 16, 16)
                if not INT_MIN <= product <= INT_MAX:
                    raise ValueError("signed-32 DOTP2 overflow")
                registers[args[2]] = signed(product, 32)
            elif op == "ADD":
                total = value(args[0]) + value(args[1])
                if not INT_MIN <= total <= INT_MAX:
                    raise ValueError("signed-32 partial/final ADD overflow")
                registers[args[2]] = total
            elif op == "SUB":
                registers[args[2]] = signed(value(args[0]) - value(args[1]), 32)
            elif op == "B":
                if args[0] not in self.labels:
                    raise ValueError(f"undefined branch label {args[0]}")
                pc = self.labels[args[0]]
            elif op == "RETURN":
                if counts["LDW"] != 0 or counts["LDDW"] != 64 or counts["DOTP2"] != 64 or counts["SUB"] != self.trip or counts["ADD"] != 64 + 64 // self.trip - 1:
                    raise ValueError(f"unexpected instruction counts: {counts}")
                if any(registers[arg] != base + 256 for arg, base in zip(self.arguments, (0x1000, 0x2010))):
                    raise ValueError("postincrement did not consume both 128-short buffers exactly")
                if set(reads) != set(memory) or any(count != 1 for count in reads.values()):
                    raise ValueError("every buffer byte must be loaded exactly once")
                return value(args[0])
        raise ValueError("kernel did not return")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clang", default="clang")
    args = parser.parse_args()
    project = args.project.resolve()
    compiler = shutil.which(args.clang)
    if not compiler:
        raise RuntimeError("clang is not installed or is not on PATH")
    kernel = LinearAssembly(project / "conv" / "cnn_dotprod128.sa")
    rng = random.Random(0x6416)
    cases: list[tuple[list[int], list[int]]] = [
        ([0] * 128, [-2672] * 128),
        ([32767] * 128, [511] * 128),
        ([-32768] * 128, [-511] * 128),
        ([200] * 128, [-32768] * 128),
        ([12128] * 125 + [0] * 3, [-1000] * 125 + [0] * 3),
        ([0, 1] * 64, [-3000, 2000] * 64),
    ]
    raw_weights = [int(x) for x in (project / "data" / "conv2_w.txt").read_text().split()]
    for channel in range(10):
        weights = raw_weights[channel::10] + [0] * 3
        for activation in (12128, 32767, -32768):
            cases.append(([activation] * 125 + [0] * 3, weights))
    for _ in range(1000):
        cases.append(([rng.randint(-12000, 12000) for _ in range(125)] + [0] * 3,
                      [rng.randint(-512, 512) for _ in range(125)] + [0] * 3))
    # Every element, including all three CNN padding positions, is exercised
    # individually with nonzero values to detect skipped/mispaired indices.
    for index in range(128):
        x = [0] * 128
        x[index] = 100 + index
        cases.append((x, [(-1 if j & 1 else 1) * (j + 1) for j in range(128)]))
    with tempfile.TemporaryDirectory(prefix="cnn-sa-semantics-") as name:
        library_path = Path(name) / "dotprod.so"
        subprocess.run([compiler, "-std=c99", "-O2", "-shared", "-fPIC", "-DUSE_LINEAR_ASM=0",
                        "-I", str(project / "conv"), str(project / "conv" / "cnn_dotprod128_c.c"),
                        "-o", str(library_path)], check=True)
        library = ctypes.CDLL(str(library_path))
        c_dot = library.cnn_dotprod128_c
        c_dot.argtypes = (ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short))
        c_dot.restype = ctypes.c_int
        buffer_type = ctypes.c_short * 128
        for index, (x, w) in enumerate(cases):
            bound = sum(abs(a * b) for a, b in zip(x, w))
            if bound > INT_MAX:
                raise ValueError(f"test case {index} violates the kernel's accumulation contract")
            expected = sum(a * b for a, b in zip(x, w))
            # Allocate enough extra bytes to select the exact bank phases.
            raw_x = ctypes.create_string_buffer(256 + 64)
            raw_w = ctypes.create_string_buffer(256 + 64)
            x_address = (ctypes.addressof(raw_x) + 31) & ~31
            w_address = ((ctypes.addressof(raw_w) + 31) & ~31) + 16
            c_x = buffer_type.from_address(x_address)
            c_w = buffer_type.from_address(w_address)
            c_x[:] = x
            c_w[:] = w
            assert x_address % 32 == 0 and w_address % 32 == 16
            c_result = c_dot(c_x, c_w)
            interpreted = kernel.execute(x, w)
            if not (expected == c_result == interpreted):
                raise ValueError(f"case {index}: Python={expected} C={c_result} interpreted .sa={interpreted}")
    print(f"PASS: {len(cases)} cases: parsed .sa data semantics == compiled C == Python scalar sum.")
    print("PASS: each call loads every byte of both 128-short buffers exactly once via 64 LDDW, with 64 DOTP2 and two safe partial sums.")
    print("LIMIT: The real .sa was parsed, NOT assembled or executed. TI assembly optimizer, ABI, instruction timing and DSP hardware remain unverified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)
