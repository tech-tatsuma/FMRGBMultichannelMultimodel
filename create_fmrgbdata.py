import os
import cv2
import numpy as np
import torch
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import zoom
import argparse

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

# 動画を読み込み、音声を抽出し、フレームを処理してテンソルとして保存する関数
def process_video(video_path, max_frames):
    # Load video
    video = cv2.VideoCapture(video_path)

    # Extract audio
    # pydubのaudiosegmentクラスを利用して、動画ファイルから音声データを抽出
    audio = AudioSegment.from_file(video_path, format="mp4")
    # 抽出した音声データをWAV形式のファイルとして出力
    audio.export("out.wav", format="wav")
    # scipyのwavfile.read関数を使用して、wavファイルを読み込む。この関数は、サンプリングレート（サンプル/秒）とサンプルデータの配列を返す。
    sample_rate, samples = wavfile.read('out.wav')

    # Get frequencies and amplitudes
    # 音声データのスペクトログラムを計算している。
    #スペクトログラムは、音声信号の周波数成分を時間的に表現したもので、音せデータの解析に広く使用される。scipyのsignal.spectrogram関数はスペクトログラムの周波数、時間、そして振幅（スペクトロ密度）を返す
    frequencies, times, amplitudes = signal.spectrogram(samples, sample_rate)

    # Iterate over each frame in the video
    frames = []
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frequencies and amplitudes to images and add to channels
        frequency_image = np.interp(frequencies, (frequencies.min(), frequencies.max()), (0, 255)).astype('uint8')
        amplitude_image = np.interp(amplitudes, (amplitudes.min(), amplitudes.max()), (0, 255)).astype('uint8')

        frame = np.dstack((frame, frequency_image, amplitude_image))

        frames.append(frame)

    frames = np.stack(frames)

    # Resize the frames to have the same length as the longest video
    num_frames = frames.shape[0]
    if num_frames < max_frames:
        zoom_factor = max_frames / num_frames
        frames = zoom(frames, [zoom_factor, 1, 1, 1], mode='nearest')

    # Save as tensor
    tensor = torch.from_numpy(frames)
    torch.save(tensor, video_path + '.pt')

# 指定されたディレクトリ内の全ての動画を処理する関数 
def process_directory(opt):
    directory_path = opt.target
    max_frames = get_max_frames(directory_path)
    for filename in os.listdir(directory_path):
        if filename.endswith('.mp4'):
            process_video(os.path.join(directory_path, filename), max_frames)

# main関数：コマンドライン引数を解析し、ディレクトリ内の動画の処理を開始する
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=str, required=True, help='folder')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning processing-----')
    process_directory(opt)
    print('-----completing processing-----')
