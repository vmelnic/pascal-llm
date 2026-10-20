"""Qualify lossless palette import against the installed ggml C decoder.

This is a source-format/capacity oracle, not a runtime or performance test.
No weights are requantized: codes are re-indexed and coefficient products
are carried in binary32. The original artifact remains read-only.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

FORMATS = {
    16: ("IQ2_XXS", 66, 32, "iq2_xxs"),
    17: ("IQ2_XS", 74, 16, "iq2_xs"),
    18: ("IQ3_XXS", 98, 32, "iq3_xxs"),
    19: ("IQ1_S", 50, 32, "iq1_s"),
    22: ("IQ2_S", 82, 16, "iq2_s"),
    29: ("IQ1_M", 56, 16, "iq1_m"),
    10: ("Q2_K", 84, 16, "q2_K"),
}
PALETTES = {
    "IQ1_S": [-9, -7, -1, 1, 7, 9],
    "IQ1_M": [-9, -7, -1, 1, 7, 9],
    "IQ2_XXS": [-43, -25, -8, 8, 25, 43],
    "IQ2_XS": [-43, -25, -8, 8, 25, 43],
    "IQ2_S": [-43, -25, -8, 8, 25, 43],
    "IQ3_XXS": [-62, -52, -44, -36, -28, -20, -12, -4,
                 4, 12, 20, 28, 36, 44, 52, 62],
}


def tables(path):
    source = path.read_text()
    result = {}
    for name, width in [("iq2xxs_grid", 8), ("iq2xs_grid", 8),
                        ("iq2s_grid", 8), ("iq3xxs_grid", 4), ("iq1s_grid", 8)]:
        text = re.search(r"GGML_TABLE_BEGIN\([^\n]*, " + name + r", [^\n]*\)\n(.*?)GGML_TABLE_END", source, re.S).group(1)
        words = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", text)]
        raw = b"".join(x.to_bytes(width, "little") for x in words)
        result[name] = np.frombuffer(raw, dtype=np.int8 if name == "iq1s_grid" else np.uint8).reshape(-1, width).astype(np.int32)
    return result


def header(f):
    def take(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
    def string(): return f.read(take("Q")).decode()
    scalar = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
    def value(kind, keep=False):
        if kind in scalar: return take(scalar[kind])
        if kind == 8: return string()
        if kind == 9:
            item, n = take("I"), take("Q")
            if item in scalar:
                f.seek(n*struct.calcsize(scalar[item]), 1)
            else:
                for _ in range(n): value(item)
            return None
        raise ValueError(kind)
    assert f.read(4) == b"GGUF" and take("I") == 3
    nt, nm = take("Q"), take("Q")
    alignment = 32
    for _ in range(nm):
        key, kind = string(), take("I")
        v = value(kind)
        if key == "general.alignment": alignment = v
    tensors = []
    for _ in range(nt):
        name, ndim = string(), take("I")
        shape = [take("Q") for _ in range(ndim)]
        tensors.append(dict(name=name, shape=shape, type=take("I"), offset=take("Q")))
    return tensors, math.ceil(f.tell()/alignment)*alignment


def reconstruct(packets, kind, group, t):
    n = len(packets)
    codes = np.empty((n, 256), dtype=np.int32)
    scales = np.empty((n, 256//group), dtype=np.float32)
    bits = packets[:, :2].copy().view("<u2").ravel()
    if kind == "IQ1_M":
        sc = packets[:, 48:56].copy().view("<u2")
        bits = (sc[:,0]>>12) | ((sc[:,1]>>8)&0xf0) | ((sc[:,2]>>4)&0xf00) | (sc[:,3]&0xf000)
    elif kind == "Q2_K":
        d = packets[:,80:82].copy().view("<f2").astype(np.float32)
        dm = packets[:,82:84].copy().view("<f2").astype(np.float32)
        coef = packets[:,:16]
        scales[:] = d * (coef & 15)
        bias = dm * (coef >> 4)
        for p in range(256):
            codes[:,p] = (packets[:,16+(p//128)*32+p%32] >> (2*((p%128)//32))) & 3
        result = scales.repeat(16, axis=1)*codes.astype(np.float32)-bias.repeat(16, axis=1)
        return result.astype(np.float32), codes, scales
    d = bits.view("<f2").astype(np.float32)
    for b in range(8):
        if kind in ("IQ2_XXS", "IQ3_XXS"):
            at = 2+b*8+4 if kind == "IQ2_XXS" else 66+4*b
            ss = packets[:,at:at+4].copy().view("<u4").ravel()
            scales[:,b] = d*((ss>>28).astype(np.float32)+np.float32(.5))*np.float32(.25 if kind == "IQ2_XXS" else .5)
        elif kind in ("IQ2_XS", "IQ2_S"):
            at = 66+b if kind == "IQ2_XS" else 74+b
            for j in range(2): scales[:,2*b+j] = d*((packets[:,at]>>(j*4)&15).astype(np.float32)+np.float32(.5))*np.float32(.25)
        elif kind == "IQ1_S":
            qh = packets[:,34+2*b:36+2*b].copy().view("<u2").ravel()
            scales[:,b] = d*(2*((qh>>12)&7)+1).astype(np.float32)*np.float32(.125)
        elif kind == "IQ1_M":
            for j in range(2): scales[:,2*b+j] = d*(2*((sc[:,b//2]>>(6*(b%2)+3*j))&7)+1).astype(np.float32)*np.float32(.125)
        for l in range(4):
            if kind == "IQ2_XXS":
                point = packets[:,2+8*b+l]
                sign7 = (ss >> (7*l)) & 127
                signs = sign7 | ((np.bitwise_count(sign7)&1)<<7)
                grid = t["iq2xxs_grid"][point]
            elif kind == "IQ2_XS":
                q = packets[:,2+2*(4*b+l):4+2*(4*b+l)].copy().view("<u2").ravel()
                sign7 = q >> 9
                signs = sign7 | ((np.bitwise_count(sign7)&1)<<7)
                grid = t["iq2xs_grid"][q&511]
            elif kind == "IQ2_S":
                point = packets[:,2+4*b+l].astype(np.int32) | (((packets[:,66+b]>>(2*l))&3).astype(np.int32)<<8)
                signs = packets[:,34+4*b+l]
                grid = t["iq2s_grid"][point]
            elif kind == "IQ3_XXS":
                sign7 = (ss >> (7*l)) & 127
                signs = sign7 | ((np.bitwise_count(sign7)&1)<<7)
                grid = np.concatenate([t["iq3xxs_grid"][packets[:,2+8*b+2*l+j]] for j in range(2)], axis=1)
            elif kind == "IQ1_S":
                point = packets[:,2+4*b+l].astype(np.int32) | (((qh>>(3*l))&7).astype(np.int32)<<8)
                delta = np.where(qh&0x8000, -1, 1)
                grid = 8*t["iq1s_grid"][point]+delta[:,None]
                signs = np.zeros(n, dtype=np.uint8)
            elif kind == "IQ1_M":
                hi = packets[:,32+2*b+l//2]
                point = packets[:,4*b+l].astype(np.int32) | (((hi>>(4*(l%2)))&7).astype(np.int32)<<8)
                delta = np.where(hi&(8<<(4*(l%2))), -1, 1)
                grid = 8*t["iq1s_grid"][point]+delta[:,None]
                signs = np.zeros(n, dtype=np.uint8)
            sign = 1-2*((signs[:,None]>>np.arange(8))&1).astype(np.int32)
            codes[:,32*b+8*l:32*b+8*l+8] = grid*sign
    palette = np.array(PALETTES[kind], dtype=np.int32)
    idx = np.searchsorted(palette,codes)
    assert np.all(idx<len(palette)) and np.array_equal(palette[idx],codes)
    # Persisted codes are indices, not approximate weight values.
    result = scales.repeat(group,axis=1)*palette[idx].astype(np.float32)
    return result, idx, scales


def main():
    ap=argparse.ArgumentParser();ap.add_argument("artifact",type=Path);ap.add_argument("--ggml",type=Path,required=True);ap.add_argument("--tables",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();tab=tables(args.tables);lib=C.CDLL(str(args.ggml))
    lib.ggml_blck_size.argtypes=[C.c_int];lib.ggml_blck_size.restype=C.c_int64
    lib.ggml_type_size.argtypes=[C.c_int];lib.ggml_type_size.restype=C.c_size_t
    records=[]; totals={}; resident=0
    edge_counts={}
    rng=np.random.default_rng(917)
    half=np.arange(65536,dtype=np.uint16)
    half=half[(half&0x7c00)!=0x7c00]
    for typ,(kind,size,group,fn) in FORMATS.items():
        observed=0
        for first in range(0,len(half),8192):
            bits=half[first:first+8192]
            packets=rng.integers(0,256,(len(bits),size),dtype=np.uint8)
            if kind=="IQ1_M":
                sc=packets[:,48:56].copy().view("<u2")
                for j in range(4): sc[:,j]=(sc[:,j]&0xfff)|(((bits>>(4*j))&15)<<12)
                packets[:,48:56]=sc.view(np.uint8)
            elif kind=="Q2_K":
                packets[:,80:82]=bits[:,None].view(np.uint8)
                packets[:,82:84]=bits[::-1,None].copy().view(np.uint8)
            else: packets[:,:2]=bits[:,None].view(np.uint8)
            actual,_,_=reconstruct(packets,kind,group,tab)
            oracle=np.empty_like(actual)
            decode=getattr(lib,"dequantize_row_"+fn)
            decode.argtypes=[C.c_void_p,C.c_void_p,C.c_int64];decode.restype=None
            decode(C.c_void_p(packets.ctypes.data),C.c_void_p(oracle.ctypes.data),oracle.size)
            mismatch=actual.view(np.uint32)!=oracle.view(np.uint32)
            assert not np.any(mismatch),("finite-half",kind,int(mismatch.sum()))
            observed+=oracle.size
        edge_counts[kind]=observed
    with args.artifact.open("rb") as f:
        tensors, start=header(f)
        for tensor in tensors:
            typ=tensor["type"]
            if typ not in FORMATS: continue
            kind, size, group, fn=FORMATS[typ]
            assert lib.ggml_blck_size(typ)==256 and lib.ggml_type_size(typ)==size
            blocks=math.prod(tensor["shape"])//256
            # Eight bounded ranges across every tensor, including its ends.
            ranges=[(int(x),min(64,blocks)) for x in np.linspace(0,max(0,blocks-64),8,dtype=np.int64)]
            raw=b""
            for offset,count in ranges:
                f.seek(start+tensor["offset"]+offset*size);raw+=f.read(count*size)
            packets=np.frombuffer(raw,dtype=np.uint8).reshape(-1,size)
            actual, codes, scales=reconstruct(packets,kind,group,tab)
            oracle=np.empty(actual.shape,dtype=np.float32)
            decode=getattr(lib,"dequantize_row_"+fn);decode.argtypes=[C.c_void_p,C.c_void_p,C.c_int64];decode.restype=None
            decode(C.c_void_p(packets.ctypes.data),C.c_void_p(oracle.ctypes.data),oracle.size)
            unequal=actual.view(np.uint32)!=oracle.view(np.uint32)
            if np.any(unequal):
                where=np.argwhere(unequal)[0];i,j=map(int,where)
                raise AssertionError((tensor["name"],kind,int(unequal.sum()),i,j,float(actual[i,j]),float(oracle[i,j]),hex(int(actual.view(np.uint32)[i,j])),hex(int(oracle.view(np.uint32)[i,j]))))
            # Canonical 16-byte row plane alignment, independently charged.
            k=tensor["shape"][0];rows=math.prod(tensor["shape"][1:])
            align=lambda n:math.ceil(n/16)*16
            stride=(align(k//2)+align(k//16)+align(k//256*4)) if kind=="Q2_K" else align(k//2)+align(k//group*4)
            charge=rows*stride;resident+=charge
            r={**tensor,"format":kind,"sample_values":oracle.size,"sample_sha256":hashlib.sha256(raw).hexdigest(),"source_bytes":blocks*size,"proposed_resident_bytes":charge,"row_stride":stride}
            records.append(r); totals[kind]=totals.get(kind,0)+oracle.size
    result={"artifact":str(args.artifact),"ggml_library":str(args.ggml.resolve()),"ggml_library_sha256":hashlib.sha256(args.ggml.read_bytes()).hexdigest(),"tables_sha256":hashlib.sha256(args.tables.read_bytes()).hexdigest(),"tensor_count":len(records),"finite_half_test_values_by_format":edge_counts,"bit_exact_sample_values":sum(totals.values()),"sample_values_by_format":totals,"proposed_missing_resident_bytes":resident,"tensors":records}
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!="tensors"}),flush=True)

if __name__=="__main__":main()
