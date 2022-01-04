# BirdResearchDBPreprocessor
## Preprocess
必要データ
- db/bird_conv_list.txt
- db/bird_name_list.txt
- db/wav/

### Scripts
```
python 00convert.py
python 01preprocess.py
python 02annotation.py
```

## Training

- Base model: https://github.com/qiuqiangkong/audioset_tagging_cnn

```
cd BirdResearchDBPreprocessor

git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git
cp -r audioset_tagging_cnn/metadata ./
CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
```

```
python main.py --max_epoch 500 --gpus 1 --batch_size 32 --learning_rate 0.001
```

