from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
filename="label01_mapping.tsv"
names=[]
for line in open(filename):
    arr=line.strip().split("\t")
    names.append(arr[1])

filename="result_cv.tsv"
true_ys=[]
pred_ys=[]
target_names=set()
for line in open(filename):
    arr=line.strip().split("\t")
    fold=arr[0]
    idx=arr[1]
    y=int(arr[2])
    pred_y=int(arr[3])
    if y!=pred_y:
        target_names.add(names[y])
        target_names.add(names[pred_y])
    pred_ys.append(pred_y)
    true_ys.append(y)

conf=confusion_matrix(true_ys,pred_ys)
eval_list=[]
for i, name in enumerate(names):
    n=sum(conf[i,:])
    pred_n=sum(conf[:,i])
    prec = conf[i,i]/sum(conf[:,i])
    rec  = conf[i,i]/sum(conf[i,:])
    f1=2/(1.0/prec+1.0/rec)
    el=[name, n, pred_n, prec, rec, f1]
    eval_list.append(el)
for el in sorted(eval_list, key=lambda x: x[1] ):
    s="\t".join(map(str,el))
    print(s)

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns

plt.figure(figsize=(40,20))
z = linkage(conf, method='single', metric='euclid')
R=dendrogram(z,labels=names)
plt.tick_params(labelsize=10)
plt.savefig(f"result.png")


g = sns.clustermap(conf, col_linkage=z, row_linkage=z,  cmap='viridis')
g.ax_col_dendrogram.set_title('Average-Corr  Ward-Euclidiean')
g.savefig('result_m0.png', dpi=150)
g = sns.clustermap(conf>0, col_linkage=z, row_linkage=z,  cmap='viridis')
g.ax_col_dendrogram.set_title('Average-Corr  Ward-Euclidiean')
g.savefig('result_m1.png', dpi=150)

print(conf)
print(set(names)-target_names)
print(len(target_names))
new_names=R["ivl"]
new_lvs=R["leaves"]
#print(new_lvs)
#print([names[i] for i in new_lvs])
#print(new_names)
temp=conf[:,new_lvs]
new_conf=temp[new_lvs,:]
#print(new_conf.shape)
#quit()


from sklearn.metrics import plot_confusion_matrix
N=10
for i in range(len(new_names)//N):
    j=i*N
    k=(i+1)*N
    plt.figure(figsize=(32,32))
    cmd = ConfusionMatrixDisplay(confusion_matrix=new_conf[j:k,j:k],display_labels=new_names[j:k])
    cmd.xticks_rotation="vertical"
    cmd.plot()
    plt.savefig(f"result{i:03d}.png")

