
import glob
import os
import subprocess

dest_dir="data_clean/"

head_arr=None
data=[]
for i in range(1,7):
    fp=open("db/No."+str(i)+".tsv")
    head=next(fp)
    head_arr=head.strip().split("\t")
    for j,line in enumerate(fp):
        arr=line.strip().split("\t")
        path="db/日本野鳥大鑑"+str(i)+"/"+arr[0]+".wav"
        if os.path.exists(path):
            data.append(["{}_{:02d}".format(i,j+1), path, str(i)]+arr)
        else:
            print("[ERROR]",path)


    
ofp=open("data.tsv","w")
head="\t".join(["audio_id","path","book_id"]+head_arr)
ofp.write(head)
ofp.write("\n")
for arr in data:
    line="\t".join(arr)
    ofp.write(line)
    ofp.write("\n")


for el in data:
    audio_id=el[0]
    path=el[1]
    dest_path=dest_dir+"/"+audio_id+".wav"
    os.makedirs(dest_dir,exist_ok=True)
    subprocess.run(["sox",path,"-b","16","-r","16000","-c","1",dest_path,"remix","2"])
    #subprocess.run(["sox",path,"-b","16","-r","16000","-c","1",dest_path])


