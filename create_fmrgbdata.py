import os
import cv2
import numpy as np
import torch
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import zoom
import argparse
import librosa
import matplotlib


# 指定されたディレクトリ内の全動画のフレーム数を調べ、最もフレーム数の多い動画のフレーム数を返す関数
def get_max_frames(directory_path):
    max_frames = float('-inf')
    for filename in os.listdir(directory_path):
        if filename.endswith('.mp4'):
            video = cv2.VideoCapture(os.path.join(directory_path, filename))
            frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if frames > max_frames:
                max_frames = frames
    return max_frames

# 音声をstereoからmonoに変換するための関数
def stereo_to_mono(audio_samples):
    return audio_samples[::2] / 2 + audio_samples[1::2] / 2

# 動画を読み込み、音声を抽出し、フレームを処理してテンソルとして保存する関数
def process_video_withmel(video_path, max_frames):
    # ビデオの読み込み
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    # mp4ファイルから音声データを読み込む
    audio = AudioSegment.from_file(video_path, format="mp4")
<<<<<<< HEAD
    
    # サンプルレートを取得
    sample_rate = audio.frame_rate
    # 音声データを配列化する
    audio_samples = np.array(audio.get_array_of_samples())
=======
    # 音声データを配列化する
    audio_samples = np.array(audio.get_array_of_samples())

>>>>>>> 491dddb996da766f003115130408ba50a5f01972
    # 音声データがstereoの時にmonoに変換する
    if audio.channels == 2:
        audio_samples = stereo_to_mono(audio_samples)

    if np.issubdtype(audio_samples.dtype, np.integer):
        audio_samples = audio_samples / np.iinfo(audio_samples.dtype).max
    elif np.issubdtype(audio_samples.dtype, np.floating):
        audio_samples = audio_samples / np.finfo(audio_samples.dtype).max

    # Generate spectrogram
    spectrogram = librosa.feature.melspectrogram(audio_samples, n_mels=128)
    db_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Resize spectrogram to match video frames
    db_spectrogram_resized = zoom(db_spectrogram, [max_frames / db_spectrogram.shape[1], 1], mode='nearest')
    db_spectrogram_resized = db_spectrogram_resized.T # Transpose to match video frames
    db_spectrogram_resized = (db_spectrogram_resized - db_spectrogram_resized.min()) / (db_spectrogram_resized.max() - db_spectrogram_resized.min()) * 255

    # 音声データの最大振幅を取得
    max_amplitude = np.max(np.abs(audio_samples))

    # ビデオのフレーム数に合わせて音声データをリサンプリングする
    resampled_audio_samples = librosa.resample(audio_samples, orig_sr=sample_rate, target_sr=int(fps))


    # Iterate over each frame in the video
    frames = []
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # RGBに変換する
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # ここからの処理で音量と周波数データを画像に変換する
        # 音声データのサンプル範囲を取得
        start_sample = int(frame_count / fps * sample_rate)
        end_sample = int((frame_count + 1) / fps * sample_rate)

        # インデックスが範囲内に収まっているか確認
        if frame_count < db_spectrogram_resized.shape[0]:
            # 音声データの範囲内での振幅を取得
            audio_frame = resampled_audio_samples[start_sample:end_sample]
            amplitude = np.max(np.abs(audio_frame))

            # 振幅を0-255の範囲にリサイズ
            resized_amplitude = (amplitude / max_amplitude) * 255
            # 振幅情報を表示
            amp_channel = np.full(frame.shape[:2], resized_amplitude)
            # Stack the channels to the frame
            frame = np.dstack((frame, freq_channel, amp_channel))
            frames.append(frame)
        else:
            print('frame_count is out of bounds')

    frames = np.stack(frames)

    # Concatenate frames and sound tensor along channel axis
    # Save as tensor
    tensor = {'frames': torch.from_numpy(frames)}
    torch.save(tensor, video_path + '.pt')

# 動画を読み込み、音声を抽出し、フレームを処理してテンソルとして保存する関数
def process_video(video_path, max_frames):
    # ビデオを読み込む
    video = cv2.VideoCapture(video_path)
<<<<<<< HEAD
    # ビデオのfpsを取得する
    fps = video.get(cv2.CAP_PROP_FPS)
    print('fps: ', fps)
    # ビデオのフレームの総数の情報を取得する
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # mp4データから音声データを取り出す
    audio = AudioSegment.from_file(video_path, format="mp4") 
    frequency_param = 100
    # 周波数を計算するための音声配列を作る作業    
    # 一秒間にfps*frequency_param個の音声データをサンプリングする
    audio = audio.set_frame_rate(int(fps*frequency_param))
    audio_sample_frequency_rate = audio.frame_rate
    audio_samples_frequency = np.array(audio.get_array_of_samples())

    # 振幅情報を取るための音声配列を作る作業
    desired_sample_rate = int(fps)
    audio = audio.set_frame_rate(desired_sample_rate)
    audio_sample_rate = audio.frame_rate
    audio_samples = np.array(audio.get_array_of_samples())

    # 音声データがstereoだった場合、monoに変換する
    if audio.channels == 2:
        audio_samples = stereo_to_mono(audio_samples)
        audio_samples_frequency = stereo_to_mono(audio_samples_frequency)
=======
    fps = video.get(cv2.CAP_PROP_FPS)
    # mp4データから音声データを取り出す
    audio = AudioSegment.from_file(video_path, format="mp4")
    # 音声データを配列化する
    audio_samples = np.array(audio.get_array_of_samples())
    # 音声データがstereoだった場合、monoに変換する
    if audio.channels == 2:
        audio_samples = stereo_to_mono(audio_samples)
>>>>>>> 491dddb996da766f003115130408ba50a5f01972

    if np.issubdtype(audio_samples.dtype, np.integer):
        audio_samples = audio_samples / np.iinfo(audio_samples.dtype).max
    elif np.issubdtype(audio_samples.dtype, np.floating):
        audio_samples = audio_samples / np.finfo(audio_samples.dtype).max

<<<<<<< HEAD
    # 周波数情報を離散フーリエ変換で取得する
    results = []
    for i in range(int(total_frames)):
        start = i * frequency_param
        end = start + frequency_param

        frequencies, times, Zxx = signal.stft(audio_samples_frequency[start:end], fs=int(fps*frequency_param))
        print('frequencies: ',frequencies)
        max_freq = np.max(np.abs(frequencies))
        results.append(max_freq)
    print('results', results[0:10])
    max_value = max(results)
    results = [(x/max_value)*255 for x in results]
  
    print('frequency:',results[0:10])
    print('length of frequencies:',len(results))

    # 音声データの最大振幅を取得
    max_amplitude = np.max(np.abs(audio_samples))
=======
    # STFTで複素数の二次元darrayを取得する
    D = librosa.stft(audio_samples)
    # spectrogram = np.abs(D)
    # 複素数を強度と位相へ変換
    S, phase = librosa.magphase(D)
    # db_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    # 強度をdb単位へ変換
    db_spectrogram = librosa.amplitude_to_db(S)

    # ビデオフレームに対応できるようにスペクトログラムのリサイズを行う
    db_spectrogram_resized = zoom(db_spectrogram, [max_frames / db_spectrogram.shape[1], 1], mode='nearest')
    db_spectrogram_resized = db_spectrogram_resized.T # Transpose to match video frames
    db_spectrogram_resized = (db_spectrogram_resized - db_spectrogram_resized.min()) / (db_spectrogram_resized.max() - db_spectrogram_resized.min()) * 255

>>>>>>> 491dddb996da766f003115130408ba50a5f01972
    # Iterate over each frame in the video
    frames = []
    frame_count = 0
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # RGBに変換する
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
<<<<<<< HEAD
        # 振幅データを取得
        amplitude = np.abs(audio_samples[frame_count])
        # 振幅を0-255の範囲にリサイズ
        resized_amplitude = (amplitude / max_amplitude) * 255
        print('resized_amplitude: ',resized_amplitude)
        # 正規化した振幅情報で画像を埋める
        amp_channel = np.full(frame.shape[:2], int(resized_amplitude))
        # 正規化したスペクトログラムで画像を埋める
        spectrogram_channel = np.full(frame.shape[:2], int(results[frame_count]))
=======

        # ここからのプログラムで音声データを画像に変換する
        f, t, Sxx = signal.spectrogram(audio_samples)
        # 現在の振幅情報を取得する
        max_amp = np.max(Sxx)
        # 正規化した振幅情報で画像を埋める
        amp_channel = np.full(frame.shape[:2], (max_amp - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx)) * 255)
        # 正規化したスペクトログラムで画像を埋める
        spectrogram_channel = np.full(frame.shape[:2], db_spectrogram_resized[int(frame_count // fps)])
>>>>>>> 491dddb996da766f003115130408ba50a5f01972
        # 5チャンネル画像を生成する
        frame = np.dstack((frame, amp_channel, spectrogram_channel))
        # frames配列に追加
        frames.append(frame)
        frame_count += 1
    # frames配列を構成する
    frames = np.stack(frames)

    # Concatenate frames and sound tensor along channel axis
    # Save as tensor
    tensor = {'frames': torch.from_numpy(frames)}
    torch.save(tensor, video_path + '.pt')

# 指定されたディレクトリ内の全ての動画を処理する関数 
def process_directory(opt):
    audio_process_method = opt.audiomethod
    directory_path = opt.target
    max_frames = get_max_frames(directory_path)
    for filename in os.listdir(directory_path):
        if filename.endswith('.mp4'):
            if audio_process_method=='simple':
                process_video(os.path.join(directory_path, filename), max_frames)
            elif audio_process_method=='mel':
                process_video_withmel(os.path.join(directory_path, filename), max_frames)
            else:
                print('音声を処理するメソッドの指定が適切ではありません')


# main関数：コマンドライン引数を解析し、ディレクトリ内の動画の処理を開始する
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=str, required=True, help='folder')
    parser.add_argument('--audiomethod',type=str, default='simple', help='audio method')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning processing-----')
    process_directory(opt)
    print('-----completing processing-----')
