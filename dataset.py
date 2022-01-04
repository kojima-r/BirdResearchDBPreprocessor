
import numpy as np
import json
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random
import os


class BirdSongDataset(Dataset):
  """Dataset class for Voice Activity Detection.
  Args:
      md_file_path (str): The path to the metadata file.
  """

  def __init__(self, label_path="label01.tsv", sample_rate=32000, segment=1):
    label_mapping={}
    data_list=[]

    label_set=set()
    for line in open(label_path):
      arr=line.strip().split("\t")
      l=arr[1]
      if l not in label_set:
        label_set.add(l)
      ##
      if os.path.exists(arr[0]):
        data_list.append((arr[0],arr[1]))
    for i,l in enumerate(sorted(list(label_set))):
      label_mapping[l]=i
    ###
    self.data_list=data_list
    self.label_mapping=label_mapping
    self.segment = segment
    self.sample_rate = sample_rate

  def __len__(self):
      return len(self.data_list)

  def __getitem__(self, idx):
      # Get the row in dataframe
      path,label = self.data_list[idx]
      
      length = len(sf.read(path)[0])
      if self.segment is not None and  length > int(self.segment * self.sample_rate):
          start = random.randint(0, length - int(self.segment * self.sample_rate))
          stop = start + int(self.segment * self.sample_rate)
      else:
          start = 0
          stop = None
      s, sr = sf.read(path, start=start, stop=stop, dtype="float32")
      if stop is None:
        n=int(self.segment * self.sample_rate)
        ss=np.zeros((n,), dtype=np.float32)
        ss[:len(s)]=s
        source = torch.from_numpy(ss)
      else:
        source = torch.from_numpy(s)
      l_idx  = self.label_mapping[label]
      ll = torch.tensor(l_idx)
      return source, ll


def from_vad_to_label(length, vad, begin, end):
    label = torch.zeros(length, dtype=torch.float)
    for start, stop in zip(vad["start"], vad["stop"]):
        label[..., start:stop] = 1
    return label[..., begin:end]

