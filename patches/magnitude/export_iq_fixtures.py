import ctypes as C
import json
import math
from pathlib import Path
import sys
import numpy as np
from gsq_palette_reference import FORMATS,header

model=Path(sys.argv[1]); lib=C.CDLL(sys.argv[2]); dest=Path(sys.argv[3]); dest.mkdir(exist_ok=True)
residents={'IQ1_S':'iq1g32','IQ1_M':'iq1g16','IQ2_XXS':'iq2g32','IQ2_XS':'iq2g16','IQ2_S':'iq2g16','IQ3_XXS':'iq3g32','Q2_K':'q2k4'}
lines=[];records=[]
with model.open('rb') as f:
    ts,start=header(f)
    for i,t in enumerate(ts):
        if t['type'] not in FORMATS:continue
        kind,size,_,fn=FORMATS[t['type']]
        nblocks=math.prod(t['shape'])//256
        # Preserve original packets across 17 sampled source rows, including
        # the tensor's end; each fixture row has two whole original packets.
        raw=b''
        for at in np.linspace(0,nblocks-2,17,dtype=np.int64):
            f.seek(start+t['offset']+int(at)*size);raw+=f.read(2*size)
        packet=np.frombuffer(raw,dtype=np.uint8)
        values=np.empty(17*512,dtype=np.float32)
        decode=getattr(lib,'dequantize_row_'+fn);decode.argtypes=[C.c_void_p,C.c_void_p,C.c_int64];decode.restype=None
        decode(C.c_void_p(packet.ctypes.data),C.c_void_p(values.ctypes.data),values.size)
        stem=f'{i:04d}-{t["name"]}'
        (dest/(stem+'.blocks')).write_bytes(raw);(dest/(stem+'.values')).write_bytes(values.tobytes())
        source='gguf_'+fn.lower()
        lines.append(f'{source}\t{residents[kind]}\t17\t512\t{stem}\n')
        records.append({**t,'stem':stem,'fixture_values':values.size})
(dest/'index.tsv').write_text(''.join(lines));(dest/'source-inventory.json').write_text(json.dumps(records,indent=2))
print('original tensor fixtures:',len(lines),'values:',len(lines)*17*512)
