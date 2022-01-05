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

## 識別学習済みモデル

必須ライブラリとして， `torch`, `torchlibrosa`　が必要になります．
学習しない場合は`pytorch-lightning`は不要です

モデルアーキテクチャは以下のリポジトリを元にしているので，このリポジトリのトップで以下のコマンドを実行し，
直下に`audioset_tagging_cnn`と`metadata`のディレクトリがある状態にする必要があります．
- Base model: https://github.com/qiuqiangkong/audioset_tagging_cnn

```
git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git
cp -r audioset_tagging_cnn/metadata ./
```

学習済みモデルを以下からダウンロードし，これを`best_models/best_models.pth`と名前を付けて保存する．
https://drive.google.com/drive/folders/18_2jTJM086wZ8m_5jNTDnqV4w7LJT5VX?usp=sharing

以上の学習済みモデルがあれば`main_deploy.py`をベースにして予測できます.
`main_deploy.py`はごく簡単なサンプルになっているので適宜書き替える必要があります．
特に，波形を読み込む部分はダミーデータになっているので，目的と状況に応じて適切に入力を作成する必要があります．

```
python main_deply.py
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

