
cnt={}
for l in open("label01.tsv"):
    arr=l.split("\t")
    k=arr[1].strip()
    if k not in cnt:
        cnt[k]=0
    cnt[k]+=1
print(cnt)

with open("label01.filter.tsv","w") as fp:
    for l in open("label01.tsv"):
        arr=l.strip().split("\t")
        k=arr[1]
        if cnt[k]>=10:
            fp.write("\t".join(arr))
            fp.write("\n")

