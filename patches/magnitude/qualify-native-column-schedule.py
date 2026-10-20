#!/usr/bin/env python3
"""Qualify selected native column schedules against the prior exact fragments."""
from pathlib import Path
import argparse,ctypes as C,importlib.util,json,time,hashlib
import numpy as np
WRAPPERS=r'''
struct Input {
    static constexpr int FACTORS=0;
    __device__ unsigned pair(const float*,unsigned m,unsigned long long k) const {
        const float x=float(int((m*127u+unsigned(k)*17u)%1021u)-510)*0.00390625f;
        const float y=float(int((m*127u+unsigned(k+1)*17u)%1021u)-510)*0.00390625f;
        return seismic_pack_bf16x2(x,y);
    }
};
template<class W> __device__ W weight() {return W{};}
template<> __device__ packets::IQ2H32 weight<packets::IQ2H32>() {
    return packets::IQ2H32(nullptr,0,packets::CodePlane{0,1},0,0);
}
template<> __device__ packets::IQ2H16 weight<packets::IQ2H16>() {
    return packets::IQ2H16(nullptr,0,packets::CodePlane{0,1},0,0);
}
template<class W> __device__ void setup(typename W::Raw &r,unsigned salt) {
    unsigned *p=reinterpret_cast<unsigned*>(&r);
    for(unsigned i=0;i<sizeof(r)/4;++i) p[i]=(salt*0x9e3779b9u+i*0x85ebca6bu)^((salt+i)<<11);
}
template<class WA,class WB> __device__ void check(unsigned *out,unsigned M) {
    const unsigned row=blockIdx.x;
    float olda[1][4]={},oldb[1][4]={},a[1][4]={},b[1][4]={};
    const WA wa=weight<WA>();const WB wb=weight<WB>();
    Input pro;Input source;
    for(unsigned kb=0;kb<7;++kb) {
        typename WA::Raw ra;typename WB::Raw rb;
        setup<WA>(ra,threadIdx.x+13*kb+1);setup<WB>(rb,threadIdx.x+19*kb+131);
        packets::Coefficients<WA::GROUPS> ca;
        packets::Coefficients<WB::GROUPS> cb;
        for(unsigned r=0;r<2;++r) {
            for(unsigned g=0;g<WA::GROUPS;++g) {
                ca.scale[r][g]=float((g+2*r+kb)%7+1)*0.00390625f;
                ca.bias[r][g]=float(g+kb)*0.001953125f;
            }
            for(unsigned g=0;g<WB::GROUPS;++g) {
                cb.scale[r][g]=float((2*g+r+kb)%7+1)*-0.0078125f;
                cb.bias[r][g]=float(r+kb)*-0.00390625f;
            }
        }
        if constexpr(WA::GROUP==WB::GROUP) {
            if(M==2) projection_reference::gemv_accumulate_simt_pair<1,2>(olda,oldb,ra,rb,wa,wb,ca,cb,M,64*kb,pro);
            else projection_reference::gemv_accumulate_simt_pair<1,0>(olda,oldb,ra,rb,wa,wb,ca,cb,M,64*kb,pro);
            projection::gemv_accumulate_simt_pair<1,0>(a,b,ra,rb,wa,wb,ca,cb,M,64*kb,pro);
        } else {
            projection_reference::gemv_accumulate_simt<1>(olda,ra,wa,ca,M,64*kb,pro);
            projection_reference::gemv_accumulate_simt<1>(oldb,rb,wb,cb,M,64*kb,pro);
            projection::gemv_accumulate_simt<1>(a,ra,wa,ca,M,64*kb,pro);
            projection::gemv_accumulate_simt<1>(b,rb,wb,cb,M,64*kb,pro);
        }
    }
    const unsigned lane=threadIdx.x%32;
    for(unsigned r=0;r<2;++r) {
        const float first=seismic_shfl_idx_f32(olda[0][2*r+row%2],4*(lane/4)+row/2);
        const float second=seismic_shfl_idx_f32(oldb[0][2*r+row%2],4*(lane/4)+row/2);
        unsigned *d=out+((row*128+threadIdx.x)*2+r)*4;
        d[0]=__float_as_uint(first);d[1]=__float_as_uint(seismic_shfl_idx_f32(a[0][2*r+row%2],4*(lane/4)+row/2));
        d[2]=__float_as_uint(second);d[3]=__float_as_uint(seismic_shfl_idx_f32(b[0][2*r+row%2],4*(lane/4)+row/2));
    }
}
struct Output {
    unsigned *out;
    __device__ void operator()(unsigned m,unsigned long long n,float a,float b) const {
        out[m*4+2*n]=__float_as_uint(a);out[m*4+2*n+1]=__float_as_uint(b);
    }
};
extern "C" __global__ void epilogue(unsigned *out) {
    if(threadIdx.x!=0)return;
    projection::GemvRowEpilogue<Output> epi{Output{out},blockIdx.x};
    epi(0,0,float(blockIdx.x)+0.5f,-float(blockIdx.x));
    epi(0,1,float(blockIdx.x)+1.5f,-float(blockIdx.x)-1.0f);
}
'''
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    spec=importlib.util.spec_from_file_location('helpers',Path(__file__).with_name('qualify-sm60-helpers.py'));h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    paths=['inference/seismic/runtime/src/native/cuda_prelude.cuh','inference/seismic/lang/native-library/cuda/element.cuh','inference/seismic/lang/native-library/cuda/packets.cuh','inference/engine/kernels/kernels/cuda/lib/projection/projection.cuh']
    texts=[(a.source_root/p).read_text() for p in paths];proj=texts[3]
    reference=(a.source_root/'selected-native-reference.cuh').read_text()
    def native(text):
        at=text.index('#if __CUDA_ARCH__ < 800');start=text.index('\n',at)+1;return text[start:text.index('#endif',start)]
    reference_native=native(reference)
    rowstart=reference.index('template <class Source> struct GemvRowSource');rowend=reference.index('// Run a GEMV warp',rowstart)
    aliases='using element::u32;using element::u64;using packets::Coefficients;using Act=element::Bf16;using Op=packets::OpBF16;\n'
    source=texts[0]+texts[1]+texts[2].replace('#include <seismic/element.cuh>','')+'\nnamespace projection_reference { '+aliases+reference_native+'\n}\nnamespace projection { '+aliases+native(proj)+reference[rowstart:rowend]+'\n}\n'+WRAPPERS
    formats=['Q2K4','Q4K','Q6K','IQ1G32','IQ1G16','IQ2H32','IQ2H16','IQ3G32','IQ4','Dense<element::Bf16>','Dense<element::F16>','Dense<element::F32>']
    pairs=[(w,w) for w in formats]+[('IQ2H32','IQ2H16'),('Q4K','Q6K')]
    for index,(wa,wb) in enumerate(pairs):source+=f'\nextern "C" __global__ void check_{index}(unsigned* out,unsigned M){{check<packets::{wa},packets::{wb}>(out,M);}}\n'
    began=time.monotonic();d=h.Device(0,source,Path('/usr/lib/x86_64-linux-gnu'));count=0
    try:
        for index,pair in enumerate(pairs):
            for rows in [2,4]:
                out=d.run(f'check_{index}',[],(rows,128,2,4),np.uint32,scalars=(C.c_uint(rows),),threads=128,grid=rows)
                selected=out[:,::4];count+=h.equal(selected[...,0],selected[...,1],f'{pair} rows{rows} first');count+=h.equal(selected[...,2],selected[...,3],f'{pair} rows{rows} second')
        out=d.run('epilogue',[],(4,4),np.uint32,threads=32,grid=4)
        expected=np.array([[row+.5,-float(row),row+1.5,-float(row)-1] for row in range(4)],np.float32).view(np.uint32);count+=h.equal(out,expected,'logical epilogue rows')
        result=dict(status='PASS',elapsed_seconds=time.monotonic()-began,exact_comparisons=count,pairs=pairs,rows=[2,4],identity=d.identity,nvrtc_version=d.nvrtc_version,source_sha256={p:hashlib.sha256(t.encode()).hexdigest() for p,t in zip(paths,texts)},cubin_sha256=d.cubin_hash,resources=d.resources);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:d.close()
if __name__=='__main__':main()
