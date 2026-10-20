#!/usr/bin/env python3
"""Qualify Magnitude's native helper ABI using installed NVRTC and CUDA driver.

This is a prerequisite gate, not a model or Pi performance qualification.
No toolkit headers, replacement weights, extra runtime or P40 allocation.
Requires NumPy; defaults to physical CUDA ordinal 0, checked as SM60.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
import numpy as np

KERNELS = r'''
extern "C" __global__ void narrow(const unsigned* bits, unsigned* out, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float x = __uint_as_float(bits[i]);
    const float y = __uint_as_float(bits[(i + 1) % count]);
    out[4*i] = seismic_f32_to_bf16(x);
    out[4*i+1] = seismic_pack_bf16x2(x,y);
    out[4*i+2] = seismic_pack_f16x2(x,y);
    out[4*i+3] = __float_as_uint(seismic_bf16_to_f32((unsigned short)bits[i]));
}
extern "C" __global__ void matrix(const unsigned short* A, const unsigned short* B,
                                  const float* initial, float* out, int bf16) {
    const unsigned lane = threadIdx.x, g = lane/4, t = lane%4;
    unsigned a[4], b[2]; float acc[4];
    for (unsigned i=0;i<4;i++) {
        const unsigned row=g+8*(i%2), k=2*t+8*(i/2);
        a[i]=(unsigned)A[row*16+k] | ((unsigned)A[row*16+k+1]<<16);
        acc[i]=initial[(g+8*(i/2))*8+2*t+i%2];
    }
    for (unsigned i=0;i<2;i++) {
        const unsigned k=2*t+8*i;
        b[i]=(unsigned)B[k*8+g] | ((unsigned)B[(k+1)*8+g]<<16);
    }
    if (bf16) seismic_mma_m16n8k16_bf16(acc,a,b);
    else seismic_mma_m16n8k16_f16(acc,a,b);
    for (unsigned i=0;i<4;i++) out[(g+8*(i/2))*8+2*t+i%2]=acc[i];
}
extern "C" __global__ void integer_matrix(const signed char* A, const signed char* B,
                                          const int* initial, int* out) {
    const unsigned lane=threadIdx.x,g=lane/4,t=lane%4;
    unsigned a[4]={},b[2]={}; int acc[4];
    for (unsigned i=0;i<4;i++) {
        const unsigned row=g+8*(i%2),k=4*t+16*(i/2);
        for (unsigned j=0;j<4;j++) a[i]|=(unsigned)(unsigned char)A[row*32+k+j]<<(8*j);
        acc[i]=initial[(g+8*(i/2))*8+2*t+i%2];
    }
    for (unsigned i=0;i<2;i++) for (unsigned j=0;j<4;j++)
        b[i]|=(unsigned)(unsigned char)B[(4*t+16*i+j)*8+g]<<(8*j);
    seismic_mma_m16n8k32_s8(acc,a,b);
    for (unsigned i=0;i<4;i++) out[(g+8*(i/2))*8+2*t+i%2]=acc[i];
}
extern "C" __global__ void fragments(const unsigned short* input,unsigned* out) {
    __shared__ __align__(16) unsigned short tile[256];
    unsigned lane=threadIdx.x;
    for (unsigned i=lane;i<256;i+=32) tile[i]=input[i];
    __syncthreads();
    const void* row=tile+(lane/8)*64+(lane%8)*8;
    unsigned x1[1],x2[2],x4[4];
    seismic_ldmatrix_x1(x1,row); seismic_ldmatrix_x2(x2,row); seismic_ldmatrix_x4(x4,row);
    out[lane*14]=x1[0];
    for (unsigned i=0;i<2;i++) out[lane*14+1+i]=x2[i];
    for (unsigned i=0;i<4;i++) out[lane*14+3+i]=x4[i];
    seismic_ldmatrix_x1_trans(x1,row); seismic_ldmatrix_x2_trans(x2,row); seismic_ldmatrix_x4_trans(x4,row);
    out[lane*14+7]=x1[0];
    for (unsigned i=0;i<2;i++) out[lane*14+8+i]=x2[i];
    for (unsigned i=0;i<4;i++) out[lane*14+10+i]=x4[i];
}
extern "C" __global__ void copies(const unsigned char* input,unsigned char* out) {
    __shared__ __align__(16) unsigned char tile[32*32];
    unsigned lane=threadIdx.x;
    seismic_cp_async_16(tile+lane*32,input+lane*16);
    seismic_cp_async_16_zfill(tile+lane*32+16,input+lane*16,lane%17);
    seismic_cp_async_commit(); seismic_cp_async_wait<1>(); seismic_cp_async_wait<0>();
    seismic_cp_async_wait_all(); __syncthreads();
    // A different thread observes each issuing thread's completed copies.
    unsigned peer=(lane+1)%32;
    for (unsigned i=0;i<32;i++) out[peer*32+i]=tile[peer*32+i];
}
extern "C" __global__ void integers(const unsigned* input,unsigned* out) {
    unsigned lane=threadIdx.x,a=input[lane],b=input[32+lane],c=input[64+lane];
    out[lane*8]=seismic_redux_add_u32(a);
    out[lane*8+1]=seismic_redux_min_u32(a); out[lane*8+2]=seismic_redux_max_u32(a);
    out[lane*8+3]=(unsigned)seismic_redux_add_s32((int)a);
    out[lane*8+4]=(unsigned)seismic_redux_min_s32((int)a);
    out[lane*8+5]=(unsigned)seismic_redux_max_s32((int)a);
    out[lane*8+6]=(unsigned)seismic_dp4a_s8((int)a,(int)b,(int)c);
    out[lane*8+7]=(unsigned)seismic_dp4a_u8s8(a,(int)b,(int)c);
}
extern "C" __global__ void loads(const unsigned* input,unsigned* out) {
    unsigned lane=threadIdx.x;
    uint4 v=seismic_ld_nc_na_v4(input+lane*4);
    seismic_prefetch_l2(input+lane*4);
    out[lane*4]=v.x; out[lane*4+1]=v.y; out[lane*4+2]=v.z; out[lane*4+3]=v.w;
}
extern "C" __global__ void hyperbolic(const float* input,float* out,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) out[i]=seismic_tanh_approx(input[i]);
}
'''


def bind(lib,name,args):
    f=getattr(lib,name); f.argtypes=args; f.restype=C.c_int
    def call(*values):
        result=f(*values)
        if result: raise RuntimeError(f'{name} returned {result}')
        return result
    return call


class Device:
    def __init__(self,ordinal,source,directory):
        self.cuda=C.CDLL('libcuda.so.1'); nv=C.CDLL(str(directory/'libnvrtc.so.12'))
        p=C.c_void_p; u=C.c_uint; n=C.c_size_t; pp=C.POINTER(p)
        bind(self.cuda,'cuInit',[u])(0)
        device=C.c_int(); bind(self.cuda,'cuDeviceGet',[C.POINTER(C.c_int),C.c_int])(C.byref(device),ordinal)
        get_attr=bind(self.cuda,'cuDeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int])
        def attr(key):
            v=C.c_int(); get_attr(C.byref(v),key,device); return v.value
        self.arch=10*attr(75)+attr(76)
        if self.arch!=60: raise RuntimeError(f'Expected SM60, found sm_{self.arch}; no fallback device')
        name=C.create_string_buffer(256)
        bind(self.cuda,'cuDeviceGetName',[p,C.c_int,C.c_int])(name,len(name),device)
        uuid=C.create_string_buffer(16)
        bind(self.cuda,'cuDeviceGetUuid_v2',[p,C.c_int])(uuid,device)
        self.identity={'ordinal':ordinal,'name':name.value.decode(),'sm':self.arch,'uuid_hex':uuid.raw.hex()}
        self.context=p(); bind(self.cuda,'cuCtxCreate_v2',[pp,u,C.c_int])(C.byref(self.context),0,device)
        self.destroy_context=bind(self.cuda,'cuCtxDestroy_v2',[p])
        self.alloc=bind(self.cuda,'cuMemAlloc_v2',[C.POINTER(C.c_uint64),n])
        self.free=bind(self.cuda,'cuMemFree_v2',[C.c_uint64])
        self.htod=bind(self.cuda,'cuMemcpyHtoD_v2',[C.c_uint64,p,n])
        self.dtoh=bind(self.cuda,'cuMemcpyDtoH_v2',[p,C.c_uint64,n])
        self.sync=bind(self.cuda,'cuCtxSynchronize',[])
        self.launch=bind(self.cuda,'cuLaunchKernel',[p,u,u,u,u,u,u,u,p,pp,pp])
        self.get_function=bind(self.cuda,'cuModuleGetFunction',[pp,p,C.c_char_p])
        major=C.c_int();minor=C.c_int(); bind(nv,'nvrtcVersion',[C.POINTER(C.c_int),C.POINTER(C.c_int)])(C.byref(major),C.byref(minor))
        self.nvrtc_version=f'{major.value}.{minor.value}'
        prog=p(); bind(nv,'nvrtcCreateProgram',[pp,C.c_char_p,C.c_char_p,C.c_int,p,p])(C.byref(prog),source.encode(),b'sm60.cu',0,None,None)
        options=[b'--gpu-architecture=sm_60',b'--std=c++17',b'--fmad=false',b'--ftz=false',b'--prec-div=true',b'--prec-sqrt=true']
        compile_f=nv.nvrtcCompileProgram; compile_f.argtypes=[p,C.c_int,C.POINTER(C.c_char_p)]; compile_f.restype=C.c_int
        code=compile_f(prog,len(options),(C.c_char_p*len(options))(*options))
        size=n(); bind(nv,'nvrtcGetProgramLogSize',[p,C.POINTER(n)])(prog,C.byref(size))
        log=C.create_string_buffer(size.value); bind(nv,'nvrtcGetProgramLog',[p,p])(prog,log)
        if code: raise RuntimeError(f'NVRTC compilation failed ({code}):\n{log.value.decode()}')
        bind(nv,'nvrtcGetCUBINSize',[p,C.POINTER(n)])(prog,C.byref(size))
        image=C.create_string_buffer(size.value); bind(nv,'nvrtcGetCUBIN',[p,p])(prog,image)
        self.cubin_hash=hashlib.sha256(image.raw).hexdigest()
        bind(nv,'nvrtcDestroyProgram',[pp])(C.byref(prog))
        self.module=p(); bind(self.cuda,'cuModuleLoadData',[pp,p])(C.byref(self.module),image)
        self.unload=bind(self.cuda,'cuModuleUnload',[p])
        self.resources={}

    def run(self,name,inputs,shape,dtype,scalars=(),threads=32,grid=1):
        output=np.empty(shape,dtype=dtype); allocations=[]
        try:
            for array in [*inputs,output]:
                if not array.flags.c_contiguous: raise ValueError('Contiguous arrays required')
                ptr=C.c_uint64(); self.alloc(C.byref(ptr),array.nbytes); allocations.append(ptr)
                if array is not output: self.htod(ptr,array.ctypes.data,array.nbytes)
            function=C.c_void_p(); self.get_function(C.byref(function),self.module,name.encode())
            attr=bind(self.cuda,'cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_void_p])
            resources={}
            for key,label in [(4,'registers_per_thread'),(1,'static_shared_bytes'),(3,'local_bytes_per_thread')]:
                val=C.c_int(); attr(C.byref(val),key,function); resources[label]=val.value
            self.resources[name]=resources
            args=[*allocations,*scalars]; pointers=(C.c_void_p*len(args))(*[C.addressof(x) for x in args])
            self.launch(function,grid,1,1,threads,1,1,0,None,pointers,None); self.sync()
            self.dtoh(output.ctypes.data,allocations[-1],output.nbytes)
            return output
        finally:
            for ptr in allocations: self.free(ptr)

    def close(self):
        self.unload(self.module); self.destroy_context(self.context)


def equal(actual,expected,label):
    ok=(actual==expected) | (np.isnan(actual)&np.isnan(expected)) if actual.dtype.kind=='f' else actual==expected
    if not np.all(ok):
        bad=np.flatnonzero(~ok.ravel())[0]
        raise AssertionError(f'{label}: mismatch at {bad}: {actual.ravel()[bad]} != {expected.ravel()[bad]}')
    return int(actual.size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prelude',type=Path,required=True)
    parser.add_argument('--nvrtc-directory',type=Path,default=Path('/usr/lib/x86_64-linux-gnu'))
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); source=args.prelude.read_text()+KERNELS
    started=time.monotonic(); dev=Device(args.device,source,args.nvrtc_directory)
    tests={}; rng=np.random.default_rng(20261009)
    try:
        # Exhaustive BF16 patterns plus each finite midpoint and either neighbor;
        # independent midpoint/parity decisions instead of the kernel's bias-and-shift.
        patterns=np.arange(65536,dtype=np.uint32)<<16
        midpoints=(patterns.astype(np.uint64)+0x8000).astype(np.uint32)
        bits=np.concatenate([patterns,midpoints-1,midpoints,midpoints+1,rng.integers(0,2**32,100000,dtype=np.uint32)])
        floats=bits.view(np.float32)
        bf_expected=np.empty(len(bits),np.uint32)
        for i,word in enumerate(bits.tolist()):
            upper=word>>16; lower=word&65535
            if (word&0x7fffffff)>0x7f800000: bf_expected[i]=0x7fff
            else: bf_expected[i]=(upper+int(lower>32768 or (lower==32768 and upper%2)))&65535
        got=dev.run('narrow',[bits],(len(bits),4),np.uint32,scalars=(C.c_uint(len(bits)),),threads=128,grid=(len(bits)+127)//128)
        tests['bf16_rne']=equal(got[:,0],bf_expected,'BF16 RNE')
        tests['bf16_pair']=equal(got[:,1],bf_expected | np.roll(bf_expected,-1)<<16,'BF16 pair')
        with np.errstate(over='ignore',invalid='ignore'):
            half=floats.astype(np.float16).view(np.uint16).astype(np.uint32)
        hp=half | np.roll(half,-1)<<16
        # NaN payloads are unspecified; require NaN classification, compare all other bits.
        for shift in (0,16):
            actual=((got[:,2]>>shift)&65535).astype(np.uint16)
            expected=((hp>>shift)&65535).astype(np.uint16)
            nan=np.isnan(expected.view(np.float16))
            tests[f'f16_pair_{shift}']=equal(actual[~nan],expected[~nan],'F16 pair finite')+equal(np.isnan(actual[nan].view(np.float16)),np.ones(np.sum(nan),bool),'F16 NaNs')
        tests['bf16_widen']=equal(got[:,3],(bits&65535)<<16,'BF16 widen')
        fmaf=C.CDLL('libm.so.6').fmaf; fmaf.argtypes=[C.c_float]*3; fmaf.restype=C.c_float
        for bf16 in (0,1):
            count=0
            for case in range(12):
                # Reference works on whole row-major matrices, not register fragments.
                a=rng.normal(size=(16,16)).astype(np.float32); b=rng.normal(size=(16,8)).astype(np.float32)
                if bf16:
                    ab=a.view(np.uint32); bb=b.view(np.uint32)
                    qa=((ab.astype(np.uint64)+0x7fff+((ab>>16)&1))>>16).astype(np.uint16)
                    qb=((bb.astype(np.uint64)+0x7fff+((bb>>16)&1))>>16).astype(np.uint16)
                    aa=(qa.astype(np.uint32)<<16).view(np.float32); bb=(qb.astype(np.uint32)<<16).view(np.float32)
                else:
                    qa=a.astype(np.float16).view(np.uint16); qb=b.astype(np.float16).view(np.uint16)
                    aa=qa.view(np.float16).astype(np.float32); bb=qb.view(np.float16).astype(np.float32)
                initial=rng.normal(size=(16,8)).astype(np.float32); expected=initial.copy()
                for row in range(16):
                    for col in range(8):
                        for k in range(16): expected[row,col]=fmaf(float(aa[row,k]),float(bb[k,col]),float(expected[row,col]))
                actual=dev.run('matrix',[qa,qb,initial],(16,8),np.float32,scalars=(C.c_int(bf16),))
                count+=equal(actual.view(np.uint32),expected.view(np.uint32),'BF16 MMA' if bf16 else 'F16 MMA')
            tests['mma_bf16' if bf16 else 'mma_f16']=count
        count=0
        for case in range(12):
            a=rng.integers(-128,128,(16,32),dtype=np.int8); b=rng.integers(-128,128,(32,8),dtype=np.int8)
            initial=rng.integers(0,2**32,(16,8),dtype=np.uint32)
            expected=(a.astype(np.int64)@b.astype(np.int64)+initial.astype(np.int64)).astype(np.uint32)
            actual=dev.run('integer_matrix',[a,b,initial],(16,8),np.uint32)
            count+=equal(actual,expected,'I8 MMA wraparound')
        tests['mma_s8_wrap']=count
        tile=rng.integers(0,65536,(4,8,8),dtype=np.uint16)
        actual=dev.run('fragments',[tile],(32,14),np.uint32)
        expected=np.empty_like(actual)
        for lane in range(32):
            for trans in (0,1):
                offset=trans*7
                for n in (1,2,4):
                    for m in range(n):
                        row,col=lane//4,2*(lane%4)
                        matrix=tile[m].T if trans else tile[m]
                        expected[lane,offset+m]=int(matrix[row,col]) | int(matrix[row,col+1])<<16
                    offset+=n
        tests['ldmatrix_x1_x2_x4_normal_trans']=equal(actual,expected,'ldmatrix')
        data=rng.integers(0,256,(32,16),dtype=np.uint8)
        actual=dev.run('copies',[data],(32,32),np.uint8)
        expected=np.zeros_like(actual); expected[:,:16]=data
        for lane in range(32): expected[lane,16:16+lane%17]=data[lane,:lane%17]
        tests['copy_all_zfill_lengths_and_peer_visibility']=equal(actual,expected,'copy')
        count=0
        for case in range(16):
            inp=rng.integers(0,2**32,(3,32),dtype=np.uint32)
            if case==0: inp[0,:8]=[0,1,0x7fffffff,0x80000000,0xffffffff,0xff000080,0x7f7f7f7f,0x80808080]
            actual=dev.run('integers',[inp],(32,8),np.uint32)
            values=inp[0].astype(np.uint64); signed=inp[0].view(np.int32).astype(np.int64)
            expected=np.empty_like(actual)
            expected[:,0]=int(values.sum())&0xffffffff; expected[:,1]=values.min(); expected[:,2]=values.max()
            expected[:,3]=int(signed.sum())&0xffffffff; expected[:,4]=int(signed.min())&0xffffffff; expected[:,5]=int(signed.max())&0xffffffff
            for lane in range(32):
                a=int(inp[0,lane]).to_bytes(4,'little'); b=int(inp[1,lane]).to_bytes(4,'little'); c=int(inp[2,lane])
                sb=[x if x<128 else x-256 for x in b]; sa=[x if x<128 else x-256 for x in a]
                expected[lane,6]=(c+sum(x*y for x,y in zip(sa,sb)))&0xffffffff
                expected[lane,7]=(c+sum(x*y for x,y in zip(a,sb)))&0xffffffff
            count+=equal(actual,expected,'redux/dp4a')
        tests['redux_dp4a_extremes_wrap']=count
        data=rng.integers(0,2**32,(32,4),dtype=np.uint32)
        tests['readonly_load_and_prefetch']=equal(dev.run('loads',[data],(32,4),np.uint32),data,'readonly load')
        xs=np.concatenate([np.linspace(-20,20,10001,dtype=np.float32),np.array([0.,-0.,np.inf,-np.inf,np.nan,np.finfo(np.float32).tiny,np.nextafter(np.float32(0),np.float32(1))],np.float32)])
        actual=dev.run('hyperbolic',[xs],xs.shape,np.float32,scalars=(C.c_uint(len(xs)),),threads=128,grid=(len(xs)+127)//128)
        # Host double tanh oracle, rounded once to F32; NVRTC tanhf is approximate.
        expected=np.tanh(xs.astype(np.float64)).astype(np.float32); finite=np.isfinite(expected)
        error=float(np.max(np.abs(actual[finite].astype(np.float64)-expected[finite].astype(np.float64))))
        if error>2**-22: raise AssertionError(f'tanh max absolute error {error}')
        equal(np.isnan(actual),np.isnan(expected),'tanh NaN')
        for i in (-7,-6,-2,-1): equal(actual[[i]].view(np.uint32),expected[[i]].view(np.uint32),'tanh signed zero/subnormal')
        tests['tanh_max_absolute_error']=error
        result={'status':'passed','prerequisite_only':True,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'prelude_sha256':hashlib.sha256(args.prelude.read_bytes()).hexdigest(),'cubin_sha256':dev.cubin_hash,'nvrtc':dev.nvrtc_version,'device':dev.identity,'tests':tests,'kernel_resources':dev.resources,'elapsed_seconds':time.monotonic()-started}
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    finally: dev.close()

if __name__=='__main__': main()
