import os
import glob

thresh = 5

data_dict={}
with open("data.tsv") as fp:
    h=next(fp)
    header=h.strip().split("\t")
    aid_index=header.index("audio_id")
    name_index=header.index("曲名")
    for line in fp:
        arr=line.strip().split("\t")
        audio_id=arr[aid_index]
        name=arr[name_index]
        data_dict[audio_id]=name


data=[]
for path in glob.glob("data01/*.wav"):
    name,ext = os.path.splitext(os.path.basename(path))
    name_list=name.split(".")
    audio_id =name_list[0]
    sep_id =name_list[1]
    l=data_dict[audio_id]
    data.append((path,l))

count={}
for path,l in data:
    if l not in count:
        count[l]=0
    count[l]+=1

ofp=open("label01_histo.tsv","w")
for k,v in sorted(count.items(), key=lambda x: x[1]):
    ofp.write("\t".join([str(k),str(v)]))
    ofp.write("\n")
    print(k,v)

ofp=open("label01.tsv","w")
for path,l in data:
    if count[l]>=thresh:
        ofp.write("\t".join([path,l]))
        ofp.write("\n")

ofp=open("label01_mapping.tsv","w")
ll=list(set([l for _,l in data if count[l]>=thresh]))
for i,l in enumerate(sorted(ll)):
    ofp.write("\t".join([str(i),l]))
    ofp.write("\n")



