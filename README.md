# dnn_kadai2

このリポジトリは、DNN実践課題で作成したプログラム・実行結果・考察用ファイルをまとめたものです。主に画像処理・コンピュータビジョンに関する深層学習手法を扱っています。

## 内容

| フォルダ | 内容 |
|---|---|
| `kadai1-autoencoder/` | AutoEncoder による画像の再構成と bottleneck feature を用いた K-means クラスタリング |
| `kadai2-pca_vgg16/` | VGG16 の fc7 特徴量を用いた PCA による次元削減と K-means クラスタリング |
| `kadai3-segmentation_detection_sam3/` | FCN-ResNet50・SAM 3 によるセマンティックセグメンテーション、および Faster R-CNN・SSD による物体検出 |

## 課題概要

### 1. AutoEncoder

COCO データセットの画像を用いて、畳み込み AutoEncoder を学習し、入力画像を低次元の bottleneck feature に圧縮してから復元しました。さらに、encoder で得られた特徴量に対して K-means クラスタリングを行い、画像がどのような特徴に基づいて分類されるかを確認しました。

主な処理内容:

- COCO 画像の読み込みと前処理
- ConvAutoEncoder の構築
- 画像の再構成結果の確認
- bottleneck feature の抽出
- K-means によるクラスタリング
- クラスタごとの画像可視化

### 2. PCA + VGG16 Feature

事前学習済み VGG16 を用いて画像から 4096 次元の fc7 特徴量を抽出し、PCA により次元削減を行いました。元の 4096 次元特徴量と、PCA 後の特徴量を用いて K-means クラスタリングを行い、次元削減がクラスタリング結果に与える影響を比較しました。

主な処理内容:

- VGG16 による画像特徴抽出
- fc7 4096 次元特徴量の取得
- PCA による次元削減
  - 累積寄与率 95%
  - 累積寄与率 90%
  - 128 次元
- K-means クラスタリング
- クラスタリング結果の比較・可視化

### 3. Segmentation / Detection / SAM 3

COCO 画像に対して、pretrained モデルを用いたセマンティックセグメンテーションと物体検出を行いました。セグメンテーションでは FCN-ResNet50 と SAM 3 を使用し、物体検出では Faster R-CNN と SSD を使用しました。

主な処理内容:

- COCO 画像の選択
- FCN-ResNet50 による semantic segmentation
- SAM 3 による prompt-based segmentation
- Faster R-CNN による object detection
- SSD による object detection
- 推論結果の可視化と比較

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
- Faster R-CNN
- SSD
- SAM 3

## 実行環境

実験は主に大学サーバ上の GPU 環境で実行しました。

例:

- OS: Linux
- Python: 3.x
- CUDA 対応 GPU 環境
- PyTorch / torchvision
- Jupyter Notebook

## 注意

大容量ファイルを避けるため、以下のようなファイルはリポジトリに含めていません。

- COCO データセット本体
- 学習済みモデルの checkpoint
- `.pt`, `.pth`, `.bin`, `.safetensors` などの大容量モデルファイル
- cache ファイル

必要なデータセットや pretrained model は、実行時に各自で準備してください。

## 作者

Oyundari Jambaldorj
