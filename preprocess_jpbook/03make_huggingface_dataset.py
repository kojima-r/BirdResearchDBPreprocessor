import datasets

import os
import glob
import pickle
import pandas as pd
import librosa
import numpy as np
import click

import shutil


# huggingface-cli login
path="./dataset01/"
@click.command()
@click.option('--input_label',     default='label01.tsv')
@click.option('--input_label_mapping',     default='label01_mapping.tsv')
@click.option('--output_path',   default=None)
@click.option('--limit_length', type=int, default=None)
def make_dataset(input_label,input_label_mapping,output_path,limit_length):
    mapping={}
    with open(input_label_mapping, 'r') as fp:
        for line in fp:
            arr=line.strip().split("\t")
            mapping[arr[1]]=arr[0]
    data=[]
    with open(input_label, 'r') as fp:
        for line in fp:
            arr=line.strip().split("\t")
            data.append((arr[0],arr[1],mapping[arr[1]]))
    os.makedirs(path+"data01/",exist_ok=True)
    with open(path+"data01/metadata.csv","w") as fp:
        fp.write(",".join(["file_name","label","description"]))
        fp.write("\n")
        for filename,label,desc in data:
            name=os.path.basename(filename)
            #shutil.copyfile(filename, "dataset01/data01/"+name)
            fp.write(",".join(map(str,[name,label,desc])))
            fp.write("\n")

    dataset = datasets.load_dataset(path, data_dir="data01")
    print(dataset["train"][0])
    dataset.push_to_hub("kojima-r/birdjpbook")


def main():
    make_dataset()

if __name__ == '__main__':
    main()


