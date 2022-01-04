import os
import glob


namelist=[]
for line in open("db/bird_name_list.txt"):
    arr=line.strip().split(" ")
    namelist.append(arr[1])
nameset = set(namelist)

nameconv={}
for line in open("db/bird_conv_list.txt"):
    arr=line.strip().split(" ")
    nameconv[arr[0]]=arr[1]

for line in open("additional_list.txt"):
    arr=line.strip().split(" ")
    nameconv[arr[0]]=arr[1]


def get_label(e):
    if e not in nameset:
        if e not in nameconv:
            print(e)
        else:
            e1=nameconv[e]
            if e1 not in nameset:
                print(e,e1)
            else:
                return e1
    else:
        return e



data=[]
for path in glob.glob("data01/*.wav"):
    name,ext1 = os.path.splitext(os.path.basename(path))
    name,ext2 = os.path.splitext(name)
    label=name.split("_")[0]
    label=label.strip('0123456789')
    l=get_label(label)
    data.append((path,l))

ofp=open("label01.tsv","w")
for path,l in data:
    ofp.write("\t".join([path,l]))
    ofp.write("\n")

count={}
for path,l in data:
    if l not in count:
        count[l]=0
    count[l]+=1

for k,v in sorted(count.items(), key=lambda x: x[1]):
    print(k,v)

