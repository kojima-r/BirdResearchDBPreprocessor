
import glob
import os
import subprocess
from multiprocessing import Pool

dest_dir="data_clean/"

os.makedirs(dest_dir,exist_ok=True)

def run(cmd):
    subprocess.run(cmd)

cmds=[]
for path in glob.glob("data/*.wav"):
    name = os.path.basename(path)
    dest_path=dest_dir+"/"+name
    cmds.append(["sox",path,"-b","16","-r","16000","-c","1",dest_path])

p = Pool(32)
p.map(run, cmds)
