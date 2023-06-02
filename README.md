# FMRGBマルチチャネルマルチモデル
音の周波数と振幅、画像のRGB情報をチャンネルとした動画データを作成し、学習を行うモデルです。
ネットワークには、3次元の畳み込みニューラルネットワークを採用しています。

## 利用方法
1. データセットの作成
```
python create_fmrgbdata.py --target <mp4データが格納されているディレクトリーパス>
```
2. データの可視化
```
python visualize.py --target <ptファイル> --result <結果を格納するディレクトリパス>
```
3. FMRGBマルチチャネルモデルの学習
```
python train.py --data <csvデータ> --epochs <学習回数> --test_size <テストデータの割合指定> --patience <早期終了パラメータ>
```
