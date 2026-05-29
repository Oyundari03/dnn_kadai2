# 課題1 AutoEncoder

## 実行ファイル

- `01_autoencoder_coco.ipynb`

## 内容

共通データセット `/export/data/dataset/COCO` を使って Denoising AutoEncoder を学習します。
入力画像にノイズを加え、出力はノイズなし画像に近づくよう MSE loss で学習します。
学習後、bottleneck feature を flatten して k=5, k=10 の k-means クラスタリングを行います。

## コードの説明

1. `COCOImageDataset` が `train2014` と `val2014` の jpg 画像を読み込みます。
2. 画像を 128x128 にリサイズし、Tensor に変換します。
3. Encoder は畳み込みと pooling で画像を bottleneck feature に圧縮します。
4. Decoder は転置畳み込みで元の画像サイズに復元します。
5. Denoising AE では入力だけに Gaussian noise を加え、損失はノイズなし画像との MSE loss で計算します。
6. 学習後、`encode_flat()` で bottleneck feature を取り出し、k-means でクラスタリングします。
7. クラスタごとの画像グリッドを `out/` に保存し、似た画像が集まるか確認します。

## 保存先

- 結果画像: `out/`
- 学習済みモデル: `model/ae_coco_denoising.pt`
- 使用データ: `/export/data/dataset/COCO`