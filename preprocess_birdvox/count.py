import glob
import soundfile as sf
sum_dur=0
for filename in glob.glob("data/*.wav"):
    f = sf.SoundFile(filename)
    frames = f.frames
    rate = f.samplerate
    duration = frames / float(rate)
    sum_dur+=duration
print(sum_dur,"sec")

