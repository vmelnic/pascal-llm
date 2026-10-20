#!/usr/bin/env python3
"""Compare paired SIMT product fragments to the unchanged independent route."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

WRAPPERS = r'''
struct Input {
    __device__ unsigned pair(const float*,unsigned m,unsigned long long k) const {
        const float x=float(int((m*127u+unsigned(k)*17u)%1021u)-510)*0.00390625f;
        const float y=float(int((m*127u+unsigned(k+1)*17u)%1021u)-510)*0.00390625f;
        return seismic_pack_bf16x2(x,y);
    }
};
template<class W> __device__ void setup(typename W::Raw& r, unsigned salt) {
    unsigned* p=reinterpret_cast<unsigned*>(&r);
    for(unsigned i=0;i<sizeof(r)/4;++i) p[i]=(salt*0x9e3779b9u+i*0x85ebca6bu)^((salt+i)<<11);
}
template<int NB,class WA,class WB>
__device__ void check(unsigned* out,unsigned M) {
    float olda[NB][4],oldb[NB][4],a[NB][4],b[NB][4];
    for(unsigned n=0;n<NB;++n) for(unsigned i=0;i<4;++i)
        olda[n][i]=a[n][i]=float(i+3*n)*0.125f,oldb[n][i]=b[n][i]=float(i+3*n)*-0.25f;
    WA wa{};WB wb{}; Input pro;
    for(unsigned kb=0;kb<7;++kb) {
        typename WA::Raw ra;typename WB::Raw rb;
        setup<WA>(ra,threadIdx.x+13*kb+1);setup<WB>(rb,threadIdx.x+19*kb+131);
        packets::Coefficients<WA::GROUPS> ca;
        packets::Coefficients<WB::GROUPS> cb;
        for(unsigned r=0;r<2;++r) for(unsigned g=0;g<WA::GROUPS;++g) {
            ca.scale[r][g]=float((g+2*r+kb)%7+1)*0.00390625f;
            cb.scale[r][g]=float((2*g+r+kb)%7+1)*-0.0078125f;
            ca.bias[r][g]=float(g+kb)*0.001953125f;
            cb.bias[r][g]=float(r+kb)*-0.00390625f;
        }
        projection::gemv_accumulate_simt<NB>(olda,ra,wa,ca,M,64*kb,pro);
        projection::gemv_accumulate_simt<NB>(oldb,rb,wb,cb,M,64*kb,pro);
        if(M==1) projection::gemv_accumulate_simt_pair<NB,1>(a,b,ra,rb,wa,wb,ca,cb,M,64*kb,pro);
        else if(M==2) projection::gemv_accumulate_simt_pair<NB,2>(a,b,ra,rb,wa,wb,ca,cb,M,64*kb,pro);
        else projection::gemv_accumulate_simt_pair<NB,0>(a,b,ra,rb,wa,wb,ca,cb,M,64*kb,pro);
    }
    for(unsigned n=0;n<NB;++n) for(unsigned j=0;j<4;++j) {
        unsigned* d=out+(threadIdx.x*NB*4+n*4+j)*4;
        d[0]=__float_as_uint(olda[n][j]);d[1]=__float_as_uint(a[n][j]);
        d[2]=__float_as_uint(oldb[n][j]);d[3]=__float_as_uint(b[n][j]);
    }
}
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('helpers',Path(__file__).with_name('qualify-sm60-helpers.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    paths=['inference/seismic/runtime/src/native/cuda_prelude.cuh',
           'inference/seismic/lang/native-library/cuda/element.cuh',
           'inference/seismic/lang/native-library/cuda/packets.cuh',
           'inference/engine/kernels/kernels/cuda/lib/projection/projection.cuh']
    texts=[(a.source_root/x).read_text() for x in paths]
    proj=texts[3];start=proj.index('template <int NB, class W, class Pro>');end=proj.index('#endif',start)
    source=texts[0]+texts[1]+texts[2].replace('#include <seismic/element.cuh>','')
    source+='\nnamespace projection {using element::u32;using element::u64;using packets::Coefficients;using Act=element::Bf16;using Op=packets::OpBF16;\n'+proj[start:end]+'\n}\n'+WRAPPERS
    formats=['Q2K4','Q4K','Q6K','IQ1G32','IQ1G16','IQ2G32','IQ2G16','IQ3G32','IQ4',
             'Dense<element::Bf16>','Dense<element::F16>','Dense<element::F32>']
    pairs=[(w,w) for w in formats]+[('Q2K4','Q6K'),('Q4K','IQ2G32'),('IQ3G32','IQ4'),('IQ4','Dense<element::Bf16>')]
    for index,(wa,wb) in enumerate(pairs):
        for nb in (1,2):
            source+=f'\nextern "C" __global__ void check_{index}_{nb}(unsigned* out,unsigned M) {{ check<{nb},packets::{wa},packets::{wb}>(out,M); }}\n'
    began=time.monotonic();d=h.Device(0,source,Path('/usr/lib/x86_64-linux-gnu'));count=0
    try:
        for index,pair in enumerate(pairs):
            for m in (1,2,3,8,11,16):
                nb=1 if m<=8 else 2
                result=d.run(f'check_{index}_{nb}',[],(128,nb,4,4),np.uint32,scalars=(C.c_uint(m),),threads=128)
                count+=h.equal(result[...,0],result[...,1],f'{pair} M={m} gate bits')
                count+=h.equal(result[...,2],result[...,3],f'{pair} M={m} up bits')
        result=dict(status='PASS',elapsed_seconds=time.monotonic()-began,exact_fragment_comparisons=count,
                    pairs=pairs,rows=[1,2,3,8,11,16],identity=d.identity,nvrtc_version=d.nvrtc_version,
                    source_sha256={path:hashlib.sha256(t.encode()).hexdigest() for path,t in zip(paths,texts)},
                    cubin_sha256=d.cubin_hash,resources=d.resources)
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:d.close()

if __name__=='__main__':main()
