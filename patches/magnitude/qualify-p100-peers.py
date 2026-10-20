#!/usr/bin/env python3
"""Qualify physical SM60 peer-copy boundaries without changing the host installation."""
import argparse
import ctypes as C
import json
import os
import time
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # This is an explicit qualification fixture for three physical P100s.
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2':
        raise SystemExit('Set CUDA_VISIBLE_DEVICES=0,1,2; P40 must be excluded')
    cuda = C.CDLL('libcuda.so.1')
    i=C.c_int; u=C.c_uint; p=C.c_void_p; ptr=C.c_uint64; size=C.c_size_t
    signatures = {
        'cuInit': [u], 'cuDeviceGetCount': [C.POINTER(i)],
        'cuDeviceGet': [C.POINTER(i),i], 'cuDeviceGetAttribute': [C.POINTER(i),i,i],
        'cuDeviceCanAccessPeer': [C.POINTER(i),i,i],
        'cuDevicePrimaryCtxRetain':[C.POINTER(p),i], 'cuDevicePrimaryCtxRelease_v2':[i],
        'cuCtxGetCurrent':[C.POINTER(p)], 'cuCtxSetCurrent':[p],
        'cuCtxEnablePeerAccess':[p,u], 'cuCtxDisablePeerAccess':[p],
        'cuStreamCreate':[C.POINTER(p),u], 'cuStreamDestroy_v2':[p], 'cuStreamSynchronize':[p],
        'cuMemAlloc_v2':[C.POINTER(ptr),size], 'cuMemFree_v2':[ptr],
        'cuMemcpyHtoD_v2':[ptr,p,size], 'cuMemcpyDtoH_v2':[p,ptr,size],
        'cuMemcpyPeerAsync':[ptr,p,ptr,p,size,p],
        'cuEventCreate':[C.POINTER(p),u], 'cuEventDestroy_v2':[p],
        'cuEventRecord':[p,p], 'cuEventSynchronize':[p],
        'cuEventElapsedTime':[C.POINTER(C.c_float),p,p],
    }
    for name,types in signatures.items():
        function=getattr(cuda,name); function.argtypes=types; function.restype=i
    def call(name,*values):
        code=getattr(cuda,name)(*values)
        if code: raise RuntimeError(f'{name}: CUDA error {code}')
    call('cuInit',0)
    count=i(); call('cuDeviceGetCount',C.byref(count)); assert count.value==3
    previous=p(); call('cuCtxGetCurrent',C.byref(previous))
    contexts=[]; streams=[]; buffers=[]; events=[]; enabled=[]
    capacity=64*1024*1024
    pattern=np.arange(capacity//4,dtype=np.uint32)*np.uint32(2654435761)
    destination=np.empty_like(pattern)
    report={'schema':'physical-peer-copy', 'devices':[], 'links':[],
            'allocation_bytes_per_device':capacity,
            'semantics':'explicit cuMemcpyPeerAsync; source and destination remain separate physical allocations',
            'integration':'standalone transfer prerequisite; no engine integration or model throughput claim'}
    try:
        for ordinal in range(3):
            device=i(); call('cuDeviceGet',C.byref(device),ordinal)
            major=i(); minor=i()
            call('cuDeviceGetAttribute',C.byref(major),75,device)
            call('cuDeviceGetAttribute',C.byref(minor),76,device)
            assert (major.value,minor.value)==(6,0)
            context=p(); call('cuDevicePrimaryCtxRetain',C.byref(context),device)
            contexts.append((device.value,context))
            call('cuCtxSetCurrent',context)
            stream=p(); call('cuStreamCreate',C.byref(stream),1); streams.append(stream)
            buffer=ptr(); call('cuMemAlloc_v2',C.byref(buffer),capacity); buffers.append(buffer)
            call('cuMemcpyHtoD_v2',buffer,pattern.ctypes.data,capacity)
            pair=[]
            for _ in range(2):
                event=p(); call('cuEventCreate',C.byref(event),0); pair.append(event)
            events.append(pair)
            report['devices'].append({'ordinal':ordinal,'sm':[major.value,minor.value]})
        for source in range(3):
            for target in range(3):
                if source==target: continue
                accessible=i()
                call('cuDeviceCanAccessPeer',C.byref(accessible),contexts[target][0],contexts[source][0])
                assert accessible.value==1
                expected=pattern+np.uint32(source+1)
                poison=np.bitwise_not(expected)
                call('cuCtxSetCurrent',contexts[source][1])
                call('cuMemcpyHtoD_v2',buffers[source],expected.ctypes.data,capacity)
                call('cuCtxSetCurrent',contexts[target][1])
                code=cuda.cuCtxEnablePeerAccess(contexts[source][1],0)
                if code==0: enabled.append((target,source))
                elif code!=704: raise RuntimeError(f'peer enable: CUDA {code}')
                link={'source':source,'target':target,'measurements':[]}
                for length in [10240,5242880,capacity]:
                    repetitions=100 if length==capacity else 300
                    call('cuMemcpyHtoD_v2',buffers[target],poison.ctypes.data,length)
                    def copy():
                        call('cuMemcpyPeerAsync',buffers[target],contexts[target][1],buffers[source],contexts[source][1],length,streams[target])
                    for _ in range(8): copy()
                    call('cuStreamSynchronize',streams[target])
                    started=time.perf_counter()
                    call('cuEventRecord',events[target][0],streams[target])
                    for _ in range(repetitions): copy()
                    call('cuEventRecord',events[target][1],streams[target])
                    call('cuEventSynchronize',events[target][1])
                    elapsed=time.perf_counter()-started
                    milliseconds=C.c_float()
                    call('cuEventElapsedTime',C.byref(milliseconds),*events[target])
                    call('cuMemcpyDtoH_v2',destination.ctypes.data,buffers[target],length)
                    assert np.array_equal(destination[:length//4],expected[:length//4])
                    device_seconds=milliseconds.value/1000
                    link['measurements'].append({'bytes':length,'repetitions':repetitions,
                        'cuda_us_per_copy':device_seconds*1e6/repetitions,
                        'wall_us_per_copy':elapsed*1e6/repetitions,
                        'cuda_GB_per_second':length*repetitions/device_seconds/1e9,
                        'exact_payload':True})
                report['links'].append(link)
                print(json.dumps(link),flush=True)
    finally:
        for target,source in reversed(enabled):
            cuda.cuCtxSetCurrent(contexts[target][1]); cuda.cuCtxDisablePeerAccess(contexts[source][1])
        for index,(device,context) in reversed(list(enumerate(contexts))):
            cuda.cuCtxSetCurrent(context)
            if index<len(events):
                for event in events[index]: cuda.cuEventDestroy_v2(event)
            if index<len(buffers): cuda.cuMemFree_v2(buffers[index])
            if index<len(streams): cuda.cuStreamDestroy_v2(streams[index])
            cuda.cuDevicePrimaryCtxRelease_v2(device)
        cuda.cuCtxSetCurrent(previous)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
