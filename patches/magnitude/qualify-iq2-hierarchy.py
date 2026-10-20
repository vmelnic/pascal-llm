#!/usr/bin/env python3
"""Qualify exact IQ2 hierarchical coefficients on the selected P100.

Exhausts all binary16 bases and all 16 source local scales, including signed
zero and nonfinite patterns, through direct and GEMM-record accessors.
"""
import argparse,importlib.util,json,hashlib,time
from pathlib import Path
import numpy as np
KERNELS=r'''
template<int G> __device__ void check(unsigned h,unsigned odd,unsigned *out) {
    float base=seismic_f16_to_f32((unsigned short)h);
    float original=(base*(float)odd)*0.125f;
    if ((h&0x7fffu)>0x7c00u) original=__uint_as_float(0x7fc00000u);
    float factor=base*0.125f;
    if ((h&0x7fffu)>0x7c00u) factor=__uint_as_float(0x7fc00000u);
    packets::IQ2Hierarchical<G> w(nullptr,0,packets::CodePlane{0,0},0,0);
    unsigned words[packets::IQ2Hierarchical<G>::SUPER_WORDS];
    for(int i=0;i<packets::IQ2Hierarchical<G>::LOCAL_WORDS;++i)words[i]=odd*0x01010101u;
    words[packets::IQ2Hierarchical<G>::LOCAL_WORDS]=__float_as_uint(factor);
    typename packets::IQ2Hierarchical<G>::Block b;
    for(int r=0;r<2;++r) {
        b.factor[r]=factor;
        for(int i=0;i<packets::IQ2Hierarchical<G>::LOCAL_WORDS;++i)b.packed[r][i]=odd*0x01010101u;
    }
    unsigned index=0;
    out[index++]=__float_as_uint(original);
    for(int kb=0;kb<4;++kb) {
        packets::Coefficients<64/G> coefficients;
        w.coefficients(b,kb,coefficients);
        for(int g=0;g<64/G;++g) {
            float scale,bias;w.staged_coefficient(words,kb,g,scale,bias);
            out[index++]=__float_as_uint(scale);
            out[index++]=__float_as_uint(coefficients.scale[0][g]);
            out[index++]=__float_as_uint(coefficients.scale[1][g]);
        }
    }
}
extern "C" __global__ void hierarchy32(unsigned *out) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=1048576)return;
    check<32>(i/16,2*(i%16)+1,out+i*25);
}
extern "C" __global__ void hierarchy16(unsigned *out) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=1048576)return;
    check<16>(i/16,2*(i%16)+1,out+i*49);
}
'''
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    helper=Path(__file__).with_name('qualify-sm60-helpers.py');spec=importlib.util.spec_from_file_location('sm60_helpers',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    paths=['inference/seismic/runtime/src/native/cuda_prelude.cuh','inference/seismic/lang/native-library/cuda/element.cuh','inference/seismic/lang/native-library/cuda/packets.cuh'];texts=[(args.source_root/p).read_text() for p in paths]
    source=texts[0]+texts[1]+texts[2].replace('#include <seismic/element.cuh>','')+KERNELS
    started=time.monotonic();dev=module.Device(0,source,Path('/usr/lib/x86_64-linux-gnu'))
    try:
        counts={}
        for group,width in [(32,25),(16,49)]:
            output=dev.run('hierarchy'+str(group),[],(1048576,width),np.uint32,threads=128,grid=8192)
            expected=np.broadcast_to(output[:,0:1],output[:,1:].shape)
            finite=((np.arange(1048576,dtype=np.uint32)//16)&0x7fff)<=0x7c00
            counts[str(group)]=module.equal(output[finite,1:],expected[finite],'exact finite/signed-zero/Inf coefficient bits')
            nan=~finite
            module.equal(np.isnan(output[nan,1:].view(np.float32)),np.ones(output[nan,1:].shape,bool),'NaN classification')
            counts[str(group)+'_canonical_nan_bits']=module.equal(output[nan,1:],expected[nan],'exact canonical NaN coefficient bits')
            counts[str(group)+'_nan_fields']=int(output[nan,1:].size)
        record={'status':'PASS','identity':dev.identity,'nvrtc_version':dev.nvrtc_version,'comparisons':counts,'elapsed_seconds':time.monotonic()-started,'source_sha256':{p:hashlib.sha256(t.encode()).hexdigest() for p,t in zip(paths,texts)},'cubin_sha256':dev.cubin_hash,'resources':dev.resources};args.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    finally:dev.close()
if __name__=='__main__':main()
