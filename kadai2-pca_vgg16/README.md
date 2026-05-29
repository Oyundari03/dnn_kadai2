# 課題2 PCA / VGG16 fc7

## 実行ファイル

- `02_pca_vgg16_fc7.ipynb`

## 内容

ImageNet 事前学習済み VGG16 から fc7 相当の 4096 次元特徴を COCO 画像200枚分抽出します。
その特徴を PCA により寄与率95%、寄与率90%、128次元へ圧縮し、それぞれ k=5, k=10 の k-means で比較します。

## コードの説明

1. `/export/data/dataset/COCO` から画像200枚を選びます。
2. VGG16 の入力形式に合わせて 224x224 にリサイズし、ImageNet mean/std で正規化します。
3. VGG16 の classifier 途中までを使い、fc7 相当の 4096 次元特徴を抽出します。
4. `PCA(n_components=0.95)` と `PCA(n_components=0.90)` で寄与率指定の圧縮を行います。
5. `PCA(n_components=128)` で固定128次元への圧縮も行います。
6. 4096次元、95%、90%、128次元の各特徴で k=5, k=10 の k-means を実行します。
7. クラスタごとの画像グリッドを `out/result` に保存して結果を比較します。

## 保存先

- PCA概要: `out/pca_summary.csv`
- 抽出特徴: `out/vgg16_fc7_100.npy`
- 使用データ: `/export/data/dataset/COCO`