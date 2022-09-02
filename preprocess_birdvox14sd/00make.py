import h5py
import glob
import soundfile as sf
import os
import numpy as np
dest_dir="data/"

os.makedirs(dest_dir,exist_ok=True)

import glob
import soundfile as sf
import os
labels={}
for filename in glob.glob("./data_org/*.h5"):
    with h5py.File(filename, "r") as container:
        value = np.array(container['sample_rate'],dtype=np.int64)
        sr=int(value)
        arr=os.path.basename(filename).split("_")
        if "X" not in arr[1]:
            name,_=os.path.splitext(os.path.basename(filename))
            print(filename,arr[1],sr)
            for i,k in enumerate(container['waveforms'].keys()):
                filepath = "data/"+name+".{:03d}.wav".format(i)
                print(k)
                labels[filepath]=arr[1]
                """
                print(container['waveforms'][k])
                w=np.array(container['waveforms'][k])
                print(w.shape,w.dtype)
                filepath = "data/"+name+".{:03d}.wav".format(i)
                _format = "WAV"
                subtype = 'FLOAT'
                sf.write(filepath, w, sr, format=_format, subtype=subtype)
                """

fp=open("label01.org.tsv","w")
for k,v in labels.items():
    fp.write("\t".join([k,v]))
    fp.write("\n")

