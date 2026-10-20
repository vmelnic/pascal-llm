#!/usr/bin/env python3
"""Check direct SIMT operands against unchanged CUDA operand decoders.

Uses installed NVRTC and one P100; no model conversion or performance claim.
Integer cases additionally use an independent host F32 reference.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

KERNELS = r'''
template <unsigned OFFSET>
__device__ void code_case(unsigned word, unsigned* out) {
    float2 old_b = packets::OpBF16::unpack(packets::OpBF16::codes<OFFSET>(word));
    float2 old_h = packets::OpF16::unpack(packets::OpF16::codes<OFFSET>(word));
    float2 direct = packets::OpSIMT<element::Bf16>::codes<OFFSET>(word);
    out[0]=__float_as_uint(old_b.x); out[1]=__float_as_uint(old_b.y);
    out[2]=__float_as_uint(old_h.x); out[3]=__float_as_uint(old_h.y);
    out[4]=__float_as_uint(direct.x); out[5]=__float_as_uint(direct.y);
}
extern "C" __global__ void codes(unsigned* out) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=16384) return;
    unsigned word=(i&127u)|((i>>7)<<16);
    code_case<0>(word,out+i*24); code_case<8>(word,out+i*24+6);
    code_case<16>(word,out+i*24+12); code_case<32>(word,out+i*24+18);
}
extern "C" __global__ void bytes(unsigned* out) {
    unsigned n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=65536) return;
    for(unsigned i=0;i<4;i++) {
        unsigned lo=(n&255u)<<(8*i),hi=(n>>8)<<(8*i);
        float2 b=packets::OpBF16::unpack(packets::OpBF16::bytes(lo,hi,i));
        float2 h=packets::OpF16::unpack(packets::OpF16::bytes(lo,hi,i));
        float2 s=packets::OpSIMT<element::Bf16>::bytes(lo,hi,i);
        unsigned* dst=out+n*24+6*i;
        dst[0]=__float_as_uint(b.x); dst[1]=__float_as_uint(b.y);
        dst[2]=__float_as_uint(h.x); dst[3]=__float_as_uint(h.y);
        dst[4]=__float_as_uint(s.x); dst[5]=__float_as_uint(s.y);
    }
}
template <class W> __device__ void packet(unsigned n,unsigned* out) {
    typename W::Raw raw;
    unsigned* words=reinterpret_cast<unsigned*>(&raw);
    for(unsigned j=0;j<sizeof(raw)/4;j++)
        words[j]=(n*0x9e3779b9u+j*0x85ebca6bu)^((n+j)<<11);
    W w{};
    for(unsigned step=0;step<4;step++) {
        unsigned b[4],h[4]; float2 sb[4],sh[4];
        w.template decode<packets::OpBF16>(raw,step,b);
        w.template decode<packets::OpF16>(raw,step,h);
        w.template decode<packets::OpSIMT<element::Bf16>>(raw,step,sb);
        w.template decode<packets::OpSIMT<element::F16>>(raw,step,sh);
        for(unsigned j=0;j<4;j++) {
            float2 ob=packets::OpBF16::unpack(b[j]),oh=packets::OpF16::unpack(h[j]);
            unsigned* dst=out+(step*4+j)*8;
            dst[0]=__float_as_uint(ob.x); dst[1]=__float_as_uint(ob.y);
            dst[2]=__float_as_uint(sb[j].x); dst[3]=__float_as_uint(sb[j].y);
            dst[4]=__float_as_uint(oh.x); dst[5]=__float_as_uint(oh.y);
            dst[6]=__float_as_uint(sh[j].x); dst[7]=__float_as_uint(sh[j].y);
        }
    }
}
extern "C" __global__ void packets_gate(unsigned* out) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=12*512) return;
    unsigned* dst=out+i*128;
    switch(i/512) {
        case 0: packet<packets::Q2K4>(i,dst); break;
        case 1: packet<packets::Q4K>(i,dst); break;
        case 2: packet<packets::Q6K>(i,dst); break;
        case 3: packet<packets::IQ1G32>(i,dst); break;
        case 4: packet<packets::IQ1G16>(i,dst); break;
        case 5: packet<packets::IQ2G32>(i,dst); break;
        case 6: packet<packets::IQ2G16>(i,dst); break;
        case 7: packet<packets::IQ3G32>(i,dst); break;
        case 8: packet<packets::IQ4>(i,dst); break;
        case 9: packet<packets::Dense<element::Bf16>>(i,dst); break;
        case 10: packet<packets::Dense<element::F16>>(i,dst); break;
        case 11: packet<packets::Dense<element::F32>>(i,dst); break;
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    helper = Path(__file__).with_name('qualify-sm60-helpers.py')
    spec = importlib.util.spec_from_file_location('sm60_helpers', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = ['inference/seismic/runtime/src/native/cuda_prelude.cuh',
             'inference/seismic/lang/native-library/cuda/element.cuh',
             'inference/seismic/lang/native-library/cuda/packets.cuh']
    texts = [(args.source_root / rel).read_text() for rel in paths]
    source = texts[0] + texts[1] + texts[2].replace('#include <seismic/element.cuh>', '') + KERNELS
    began = time.monotonic()
    dev = module.Device(0, source, Path('/usr/lib/x86_64-linux-gnu'))
    try:
        codes = dev.run('codes', [], (16384, 4, 6), np.uint32, threads=128, grid=128)
        n = np.arange(16384, dtype=np.uint32)
        vals = np.stack([n & 127, n >> 7], axis=1).astype(np.float32)
        for index, offset in enumerate([0, 8, 16, 32]):
            expected = (vals - np.float32(offset)).view(np.uint32)
            for col in [0, 2, 4]:
                module.equal(codes[:, index, col:col+2], expected, 'unsigned codes')
        bytes_out = dev.run('bytes', [], (65536, 4, 6), np.uint32, threads=128, grid=512)
        n = np.arange(65536, dtype=np.uint32)
        vals = np.stack([n & 255, n >> 8], axis=1).astype(np.int16)
        vals = np.where(vals >= 128, vals - 256, vals).astype(np.float32).view(np.uint32)
        for index in range(4):
            for col in [0, 2, 4]:
                module.equal(bytes_out[:, index, col:col+2], vals, 'signed bytes')
        packets = dev.run('packets_gate', [], (12, 512, 16, 8), np.uint32, threads=128, grid=48)
        module.equal(packets[..., :2], packets[..., 2:4], 'BF16 packet operand bits')
        module.equal(packets[..., 4:6], packets[..., 6:8], 'F16 packet operand bits')
        result = dict(status='PASS', elapsed_seconds=time.monotonic()-began,
                      identity=dev.identity, nvrtc_version=dev.nvrtc_version,
                      exact_packet_comparisons=12*512*16*4,
                      integer_reference_comparisons=16384*4*6+65536*4*6,
                      source_sha256={rel: hashlib.sha256(text.encode()).hexdigest()
                                     for rel, text in zip(paths, texts)},
                      cubin_sha256=dev.cubin_hash, resources=dev.resources)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    finally:
        dev.close()


if __name__ == '__main__':
    main()
