
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
import glob

fs=16000
frame_len = int(100 * fs /1000) # 20ms
frame_shift = int(5 * fs /1000) # 10ms
thresh_u=0.01
thresh_interval=300 #msec
out_plot_dir="data01_plot/"
out_dir="data01/"
os.makedirs(out_plot_dir,exist_ok=True)
os.makedirs(out_dir,exist_ok=True)
expand_sec=0.05

n_jobs=64
verbose=True
filelist=list(glob.glob("data_clean/*.wav"))


def get_boundaries(frame_idxs):
    if len(frame_idxs)==0:
        return None, None
    start_idxs = [frame_idxs[0]]
    end_idxs = []

    shapeofidxs = np.shape(frame_idxs)
    for i in range(shapeofidxs[0]-1):
        if (frame_idxs[i + 1] - frame_idxs[i]) != 1:
            end_idxs.append(frame_idxs[i]+1)
            start_idxs.append(frame_idxs[i+1])

    end_idxs.append(frame_idxs[-1])
    if end_idxs[-1] == start_idxs[-1]:
        end_idxs.pop()
        start_idxs.pop()
    assert len(start_idxs) == len(end_idxs), 'Error! Num of start_idxs doesnt match Num of end_idxs.'
    start_idxs = np.array(start_idxs)
    end_idxs = np.array(end_idxs)
    start_t = start_idxs * frame_shift / fs
    end_t = end_idxs * frame_shift / fs
    return start_t, end_t

def union_intervals(seg,thresh_interval):
  new_s=None
  prev_e=None
  new_segment=[]
  for s,e in seg:
    if prev_e is not None:
      if (s-prev_e)*1000 <= thresh_interval:
        prev_e=e
      else:
        new_segment.append((new_s,prev_e))
        prev_e=e
        new_s=s
    else:
      new_s=s
      prev_e=e
  if prev_e is not None:
    new_segment.append((new_s,prev_e))
  return new_segment

def expand_intervals(new_segment,sec=0.1):
  new_segment2=[]
  for s, e in new_segment:
    s=s-sec
    e=e+sec
    if s<0:
      s=0
    new_segment2.append((s,e))
  return new_segment2


def plot_segmentation(path,seg,wave):
  plt.figure(figsize=(32, 4))
  t = np.linspace(0, len(wave)/fs, len(wave))
  plt.plot(t, wave, label='Waveform')
  for s, e in seg:
    ms=(e-s)*1000
    if ms>10:
      plt.axvline(x=s, color='#d62728') # red vertical line
      plt.axvline(x=e, color='#2ca02c') # green vertical line
  plt.legend(loc='best')
  plt.savefig(path)
  plt.close()

def plot_histogram(path,u):
    plt.figure(figsize=(5, 5))
    n, bins, patches = plt.hist(u.T, 100, facecolor='g', alpha=0.75)
    plt.savefig(path)
    plt.close()


def preprocess(filename):
    print(filename)
    name,ext=os.path.splitext(os.path.basename(filename))
    est1 = sf.read(filename)[0]
    wave, index = librosa.effects.trim(est1, top_db=25)

    u=librosa.feature.rms(wave,frame_length=frame_len,hop_length=frame_shift)
    #v=librosa.feature.zero_crossing_rate(wave, frame_length=frame_len, hop_length=frame_shift, threshold=0)
    #v=v[0]
    plot_histogram(out_plot_dir+name+".histo.png",u)
    u=u[0]
    frame_idxs = np.where(u > thresh_u)[0]


    start_t, end_t = get_boundaries(frame_idxs)
    if start_t is None:
        print("[ERROR]",filename)
        return
    path=out_plot_dir+name+".seg00.png"
    plot_segmentation(path,zip(start_t, end_t),wave)

    new_segment=union_intervals(zip(start_t, end_t),thresh_interval)
    path=out_plot_dir+name+".seg01.png"
    plot_segmentation(path,new_segment,wave)

    new_segment2=expand_intervals(new_segment,expand_sec)
    path=out_plot_dir+name+".seg02.png"
    plot_segmentation(path,new_segment2,wave)

    for i,seg in enumerate(new_segment2):
        s, e = seg
        si=int(s*fs)
        ei=int(e*fs)
        w=wave[si:ei]
        path=out_dir+name+".{:04d}.wav".format(i)
        sf.write(path, w, fs)


from multiprocessing import Pool
p = Pool(64)
p.map(preprocess,filelist)

"""
with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
    parallel(delayed(preprocess)(filename) for filename in tqdm(filelist,total=len(filelist), leave=False))
"""
