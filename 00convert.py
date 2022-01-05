
import glob
import os
import subprocess

dest_dir="data_clean/"

os.makedirs(dest_dir,exist_ok=True)


for path in glob.glob("db/wav/*.wav"):
    name = os.path.basename(path)
    dest_path=dest_dir+"/"+name
    subprocess.run(["sox",path,"-b","16","-r","16000","-c","1",dest_path])


