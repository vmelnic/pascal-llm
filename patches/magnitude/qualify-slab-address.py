#!/usr/bin/env python3
"""Check slab addresses against independent integer arithmetic on SM60."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

KERNEL = r'''
extern "C" __global__ void addresses(const unsigned* table,
    const unsigned long long* indices, unsigned long long* out,
    unsigned long long count, unsigned long long rows,
    unsigned long long bytes) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        out[i] = reinterpret_cast<unsigned long long>(
            slab::row(table, indices[i], rows, bytes));
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    helper = Path(__file__).with_name('qualify-sm60-helpers.py')
    spec = importlib.util.spec_from_file_location('helpers', helper)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    path = args.source_root / 'inference/seismic/lang/native-library/cuda/slab.cuh'
    source = path.read_text()
    dev = h.Device(0, source + KERNEL, Path('/usr/lib/x86_64-linux-gnu'))
    began = time.monotonic()
    try:
        # Valid pointer-table slots; deliberately exercise both pointer words.
        bases = (np.arange(4096, dtype=np.uint64) * 0x123456780
                 + np.uint64(0x1000000000))
        table = bases.view(np.uint32)
        checks = 0
        for rows in [1, 2, 3, 4, 17, 256]:
            indices = np.arange(4096 * rows, dtype=np.uint64)
            for row_bytes in [0, 16, 3741568, 1 << 34]:
                out = dev.run('addresses', [table, indices], indices.shape,
                              np.uint64, scalars=[C.c_uint64(indices.size),
                                  C.c_uint64(rows), C.c_uint64(row_bytes)],
                              threads=128, grid=(indices.size + 127) // 128)
                expected = bases[indices // rows] + (indices % rows) * row_bytes
                checks += h.equal(out, expected, 'slab address bits')
        result = dict(status='PASS', comparisons=checks,
                      elapsed_seconds=time.monotonic() - began,
                      identity=dev.identity, nvrtc_version=dev.nvrtc_version,
                      source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      cubin_sha256=dev.cubin_hash, resources=dev.resources)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        dev.close()


if __name__ == '__main__':
    main()
