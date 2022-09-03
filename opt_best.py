import glob
import json
import os

result=[]
for path in glob.glob("study_100/trial*/result_valid.json"):
    fp=open(path)
    obj=json.load(fp)
    if "map" in obj:
        d=os.path.dirname(path)
        result.append((obj["map"],d))

for sc,d in sorted(result):
    print("{:2.3f}: {}".format(sc,d))

