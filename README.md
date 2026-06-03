# dnn_kadai2

このリポジトリは、DNN実践課題で作成したプログラム、実行結果、考察用ファイルをまとめたものです。主に画像処理・コンピュータビジョン・自然言語処理に関する深層学習手法を扱っています。

## 内容

| 課題 | 内容 |
|---|---|
| `kadai1` | AutoEncoder による画像再構成と特徴量クラスタリング |
| `kadai2` | VGG16 特徴量に対する PCA と K-means クラスタリング |
| `kadai3` | セマンティックセグメンテーション・SAM 3 による画像分割 |
| `kadai4` | Faster R-CNN / SSD による物体検出 |
| `kadai5-word2vec` | Word2Vec による単語ベクトル演算と類似語検索 |
| `kadai6` | Vision Transformer による画像分類 |
| `kadai7-caption_generation` | BLIP を用いた画像キャプション生成 |
| `kadai12-clip_zero_shot` | CLIP を用いた zero-shot 画像分類 |

## 課題概要

### kadai1: AutoEncoder

COCO データセットの画像を用いて、畳み込み AutoEncoder を学習しました。入力画像を encoder により低次元の bottleneck feature に圧縮し、decoder により元画像を再構成しました。

また、encoder から得られた特徴量に対して K-means クラスタリングを行い、画像がどのような特徴に基づいてグループ化されるかを確認しました。

主な処理内容:

- COCO 画像の読み込みと前処理
- ConvAutoEncoder の構築
- 画像の再構成結果の確認
- bottleneck feature の抽出
- K-means によるクラスタリング
- クラスタごとの画像可視化

### kadai2: PCA + VGG16 Feature

事前学習済み VGG16 を用いて画像から 4096 次元の fc7 特徴量を抽出し、PCA によって次元削減を行いました。

元の 4096 次元特徴量と PCA 後の特徴量を用いて K-means クラスタリングを行い、次元削減がクラスタリング結果に与える影響を比較しました。

主な処理内容:

- VGG16 による画像特徴抽出
- fc7 4096 次元特徴量の取得
- PCA による次元削減
  - 累積寄与率 95%
  - 累積寄与率 90%
  - 128 次元
- K-means クラスタリング
- クラスタリング結果の比較・可視化

### kadai3: Segmentation / SAM 3

COCO 画像に対して、pretrained モデルを用いたセマンティックセグメンテーションを行いました。

FCN-ResNet50 と SAM 3 を使用し、画像中の対象物領域を分割しました。従来の semantic segmentation モデルと、prompt に基づいて柔軟な分割が可能な SAM 3 の結果を比較しました。

主な処理内容:

- COCO 画像の選択
- FCN-ResNet50 による semantic segmentation
- SAM 3 による prompt-based segmentation
- 分割結果の可視化と比較

### kadai4: Detection: Faster R-CNN / SSD

COCO 画像に対して、pretrained object detection モデルを用いた物体検出を行いました。

Faster R-CNN と SSD を使用し、画像中の物体のクラスラベル、信頼度スコア、bounding box を出力しました。2つの検出モデルの結果を比較し、検出精度や検出される物体の違いを確認しました。

主な処理内容:

- COCO 画像の読み込み
- Faster R-CNN による object detection
- SSD による object detection
- 検出結果の bounding box 描画
- score threshold による検出結果の絞り込み
- モデルごとの検出結果の比較

### kadai5: Word2Vec

Word2Vec を用いて単語をベクトルとして表現し、単語間の意味的な関係を確認しました。単語ベクトルの加減算により、類似語検索や意味関係の推定を行いました。

主な処理内容:

- 学習済み Word2Vec モデルの読み込み
- 単語ベクトルの取得
- 類似単語の検索
- ベクトル演算による意味関係の確認

実験例:

- `king - man + woman ≒ queen`
- `paris + japan - france ≒ tokyo`
- `tokyo + france - japan ≒ paris`
- `walking + swim - walk ≒ swimming`

### kadai6: Vision Transformer

Vision Transformer を用いて画像分類を行いました。CNN とは異なり、画像を小さな patch に分割し、それぞれを token として Transformer に入力することで画像全体の特徴を学習します。

本課題では、Vision Transformer の仕組みを確認し、画像分類における Transformer ベースモデルの特徴を考察しました。

主な処理内容:

- 画像の patch 分割
- Vision Transformer モデルの利用
- 画像分類の実行
- CNN ベースモデルとの考え方の違いの確認
- 分類結果の確認と考察

### kadai7: Caption Generation

BLIP を用いて、COCO 画像に対する画像キャプション生成を行いました。画像を入力として与え、モデルが画像内容を自然言語の文章として出力できるかを確認しました。

主な処理内容:

- COCO 画像の読み込み
- BLIP による image captioning
- 生成された caption の保存
- 画像内容と生成 caption の比較

生成結果の例:

- `person`: a man riding a motorcycle down a street
- `bus`: a bus on the road
- `airplane`: a plane flying over a church in a town
- `dog`: a dog laying in a dirty bathroom
- `cat`: a cat standing on a toilet

### kadai12: CLIP Zero-Shot Classification

CLIP は画像エンコーダとテキストエンコーダを同じ embedding 空間へ写像する Vision-Language Model です。画像特徴量と text prompt の特徴量の類似度を計算することで、追加学習なしで zero-shot 画像分類を行いました。

主な処理内容:

- COCO 画像の読み込み
- text prompt の作成
- CLIP による画像特徴量・テキスト特徴量の抽出
- cosine similarity による類似度計算
- zero-shot 画像分類結果の保存

実験例:

- `a photo of a person`
- `a photo of a bus`
- `a photo of an airplane`

## 使用技術

- Python
- PyTorch
- torchvision
- scikit-learn
- NumPy
- Matplotlib
- OpenCV
- COCO Dataset
- VGG16
- FCN-ResNet50
- SAM 3
- Faster R-CNN
- SSD
- Word2Vec
- Vision Transformer
- BLIP
- CLIP

## 実行環境

実験は主に大学サーバ上の GPU 環境で実行しました。

例:

- OS: Linux
- Python: 3.x
- CUDA 対応 GPU 環境
- PyTorch / torchvision
- Jupyter Notebook
- Miniconda / conda 仮想環境

## 注意

大容量ファイルを避けるため、以下のようなファイルはリポジトリに含めていません。

- COCO データセット本体
- dataset cache
- 学習済みモデルの checkpoint
- `.pt`, `.pth`, `.bin`, `.safetensors` などの大容量モデルファイル
- Hugging Face cache
- Jupyter の `.ipynb_checkpoints`

必要なデータセットや pretrained model は、実行時に各自で準備してください。

## 作者

Oyundari Jambaldorj
