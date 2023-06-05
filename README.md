# FMRGBマルチチャネルマルチモデル
音の周波数と振幅、画像のRGB情報をチャンネルとした動画データを作成し、学習を行うモデルです。
ネットワークには、3次元の畳み込みニューラルネットワークを採用しています。

## 利用方法
1. データセットの作成
```
python create_fmrgbdata.py --target <mp4データが格納されているディレクトリーパス> --audiomethod <simple or mel>
```
target: fmrgbデータに変換したい動画データが格納されているディレクトリの指定

audiometho: 音声データの解析をシンプルなスペクトログラムで解析するかメロスペクトログラムで解析するかをしてするオプション

### 音声データの画像化
スペクトログラム

1. STFT（Short Time Fourier Transform）の計算：librosa.stft(audio_samples)関数を使って、短時間フーリエ変換を行います。これは、音声信号を小さなチャンク（フレーム）に分割し、各フレームにフーリエ変換を適用して複素数の結果を生成します。

2. スペクトログラムの計算：np.abs(D)を使って、STFTの結果から振幅スペクトログラムを計算します。これは、各フレームと周波数で音声信号の振幅を示します。

3. デシベルスケールへの変換：librosa.amplitude_to_db(spectrogram, ref=np.max)を使って、振幅スペクトログラムをデシベルスケールに変換します。デシベルスケールは、人間の感覚により適した尺度で音声信号のダイナミックレンジを表現します。

4. スペクトログラムのリサイズ：zoom(db_spectrogram, [max_frames / db_spectrogram.shape[1], 1], mode='nearest')を使って、スペクトログラムをリサイズします。ここで、max_framesは動画のフレーム数で、スペクトログラムを動画のフレーム数に合わせてリサイズします。

5. スペクトログラムの正規化：db_spectrogram_resized = (db_spectrogram_resized - db_spectrogram_resized.min()) / (db_spectrogram_resized.max() - db_spectrogram_resized.min()) * 255 これは0-255の範囲に正規化します。こうすることで、各値が画像ピクセルとして扱えるようになります。これは画像処理タスクで一般的に行われるステップで、データを扱いやすくします。

メルスペクトログラム

1. 音声抽出：まず、AudioSegment.from_file(video_path, format="mp4")を用いて動画ファイルから音声を抽出しています。その後、音声サンプルをnp.array(audio.get_array_of_samples())で配列に変換し、音声サンプルを正規化しています。

2. メルスペクトログラムの生成：librosa.feature.melspectrogram(audio_samples, n_mels=128)を用いて音声信号からメルスペクトログラムを生成しています。メルスペクトログラムは、音声信号の周波数領域をメルスケールで表示したもので、人間の聴覚特性を反映しています。

3. デシベルスケールへの変換：librosa.power_to_db(spectrogram, ref=np.max)を用いて、メルスペクトログラムをデシベルスケールに変換しています。デシベルスケールは、人間の聴覚が対数的であるという事実を反映しています。

4. メルスペクトログラムのリサイズ：zoom(db_spectrogram, [max_frames / db_spectrogram.shape[1], 1], mode='nearest')を用いて、メルスペクトログラムをリサイズしています。ここで、max_framesは動画のフレーム数で、メルスペクトログラムを動画のフレーム数に合わせてリサイズします。

5. メルスペクトログラムの正規化と画像化：最後に、db_spectrogram_resized = (db_spectrogram_resized - db_spectrogram_resized.min()) / (db_spectrogram_resized.max() - db_spectrogram_resized.min()) * 255を用いて、メルスペクトログラムを0-255の範囲に正規化しています。こうすることで、各値が画像ピクセルとして扱えるようになり、画像化されます。

### データの長さの揃える方法
1. ディレクトリ内の全ての動画を調査し、フレーム数が最も多い動画のフレーム数を取得します。

2. その後、各動画を処理する際に動画のフレーム数が最大フレーム数より少ない場合に、フレーム数を最大フレーム数にリサイズします。リサイズにはscipy.ndimage.zoom関数を用います。この関数は、入力配列の各次元を特定のズーム因子に基づいて拡大または縮小します。ズーム因子は、現在のフレーム数を最大フレームで割ったもので計算されます。
    
2. データの可視化
```
python visualize.py --target <ptファイル> --result <結果を格納するディレクトリパス>
```
3. FMRGBマルチチャネルモデルの学習
```
python train.py --data <csvデータ> --epochs <学習回数> --test_size <テストデータの割合指定> --patience <早期終了パラメータ>
```
