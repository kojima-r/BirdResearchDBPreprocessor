cnt={}
for l in open("label01.org.tsv"):
    arr=l.split("\t")
    k=arr[1].strip()
    if k[0]=="1" and "0" not in k:
        if k not in cnt:
            cnt[k]=0
        cnt[k]+=1
print(cnt)

with open("label01_mapping.tsv","w") as fp:
    for i,k in enumerate(sorted(cnt.keys())):
        fp.write("\t".join([str(i),k]))
        fp.write("\n")

with open("label01.filter.tsv","w") as fp:
    for l in open("label01.tsv"):
        arr=l.strip().split("\t")
        k=arr[1]
        if k in cnt and  cnt[k]>=10:
            fp.write("\t".join(arr))
            fp.write("\n")

