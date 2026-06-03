# dnn_kadai2

このリポジトリは、DNN実践課題で作成したプログラム、実行結果、考察用ファイルをまとめたものです。主に画像処理・コンピュータビジョン・自然言語処理に関する深層学習手法を扱っています。

## 内容

| 課題 | 内容 |
|---|---|
| `kadai1` | AutoEncoder による画像再構成と特徴量クラスタリング |
| `kadai2` | VGG16 特徴量に対する PCA と K-means クラスタリング |
| `kadai3` | セマンティックセグメンテーション・物体検出・SAM 3 による画像分割 |
| `kadai4` | マルチ GPU を用いた CNN / ResNet の学習実験 |
| `kadai5` | Word2Vec による単語ベクトル演算と類似語検索 |
| `kadai6` | 画像分類モデルの判断根拠の可視化（Occlusion / Saliency / Grad-CAM など） |
| `kadai7` | Stable Diffusion を用いた画像生成・img2img・inpainting |
| `kadai12` | CLIP / BLIP などの Vision-Language Model を用いた画像理解 |

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

### kadai3: Segmentation / Detection / SAM 3

COCO 画像に対して、pretrained モデルを用いたセマンティックセグメンテーションと物体検出を行いました。

セグメンテーションでは FCN-ResNet50 と SAM 3 を使用し、物体検出では Faster R-CNN と SSD を使用しました。

主な処理内容:

- COCO 画像の選択
- FCN-ResNet50 による semantic segmentation
- SAM 3 による prompt-based segmentation
- Faster R-CNN による object detection
- SSD による object detection
- 推論結果の可視化と比較

### kadai4: Multi-GPU CNN / ResNet Training

CIFAR-10 データセットを用いて、ResNet152 による画像分類実験を行いました。1 GPU、2 GPU、4 GPU の場合で学習時間と精度を比較し、マルチ GPU による学習速度の変化を確認しました。

主な処理内容:

- CIFAR-10 データセットの読み込み
- ResNet152 による画像分類
- DataParallel を用いたマルチ GPU 学習
- 1 GPU / 2 GPU / 4 GPU の比較
- 学習時間と accuracy の比較

実験結果の例:

| GPU 数 | 学習時間 | Accuracy |
|---|---:|---:|
| 1 GPU | 545.1 sec | 65.86% |
| 2 GPU | 308.9 sec | 61.62% |
| 4 GPU | 169.5 sec | 58.22% |

マルチ GPU により学習時間は短縮されましたが、batch size の増加などの影響により accuracy はやや低下しました。

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

### kadai6: Visualization of CNN Decision

CNN の判断根拠を可視化するために、Occlusion、Saliency Map、Guided Backpropagation、Grad-CAM などの手法を用いました。

画像分類モデルが画像中のどの領域を重視して分類しているかを確認し、モデルの認識過程を考察しました。

主な処理内容:

- 事前学習済み CNN モデルの利用
- 入力画像の分類
- Occlusion による重要領域の確認
- Saliency Map による勾配可視化
- Guided Backpropagation によるエッジ・特徴の可視化
- Grad-CAM による注目領域の可視化

### kadai7: Stable Diffusion

Stable Diffusion を用いて、テキストから画像を生成する text-to-image、画像を別の画像へ変換する img2img、画像の一部を編集する inpainting を行いました。

プロンプトを変更することで生成結果がどのように変化するかを確認し、拡散モデルによる画像生成の特徴を考察しました。

主な処理内容:

- text-to-image による画像生成
- img2img による画像変換
- inpainting による部分編集
- prompt の違いによる生成結果の比較
- Stable Diffusion v1.5 / SDXL-Turbo などの利用

実験例:

- cat image generation
- cat → dog の img2img
- apple → banana の img2img
- cat → golden retriever の inpainting
- desert landscape → snowy landscape の変換

### kadai12: Vision-Language Model

画像とテキストを同時に扱う Vision-Language Model を用いて、画像理解に関する実験を行いました。

CLIP では画像とテキストを同じ embedding 空間に写像し、text prompt との類似度を計算することで zero-shot 画像分類を行いました。BLIP では画像から caption を生成し、画像内容を自然言語で説明しました。

主な処理内容:

- CLIP による zero-shot 画像分類
- text prompt と画像特徴量の類似度計算
- BLIP による image captioning
- COCO 画像を用いた推論
- 画像とテキストの対応関係の考察

実験例:

- `a photo of a person`
- `a photo of a bus`
- `a photo of an airplane`
- BLIP による COCO 画像の caption 生成

## 使用技術

- Python
- PyTorch
- torchvision
- scikit-learn
- NumPy
- Matplotlib
- OpenCV
- COCO Dataset
- CIFAR-10
- VGG16
- ResNet152
- FCN-ResNet50
- Faster R-CNN
- SSD
- SAM 3
- Word2Vec
- Stable Diffusion
- CLIP
- BLIP

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
- CIFAR-10 などの dataset cache
- 学習済みモデルの checkpoint
- `.pt`, `.pth`, `.bin`, `.safetensors` などの大容量モデルファイル
- Hugging Face cache
- Jupyter の `.ipynb_checkpoints`

必要なデータセットや pretrained model は、実行時に各自で準備してください。

## 作者

Oyundari Jambaldorj
