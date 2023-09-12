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
import matplotlib.pyplot as pyplot
from PIL import Image
import io
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import resample
import datetime
import sys


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
def process_video_withmel(video_path, max_frames, parameter, isfmrgb, skip, output_dir):
    # ビデオを読み込む
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ビデオのfpsを取得する
    fps = video.get(cv2.CAP_PROP_FPS)
    print('fps: ', fps)
    print(isfmrgb)

    # flowoutput動画を読み込む
    flow_output_filename = os.path.splitext(os.path.basename(video_path))[0] + 'flowoutput.mp4'
    print('flow_output_filename:',flow_output_filename)
    flow_output_path = os.path.join(os.path.dirname(video_path), flow_output_filename)
    print('flow_output_path:',flow_output_path)
    flow_video = cv2.VideoCapture(flow_output_path)

    # ビデオのフレームの総数の情報を取得する
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    try:
        # mp4データから音声データを取り出す
        audio = AudioSegment.from_file(video_path, format="mp4") 
    except:
        print(f"Could not extract audio from {video_path}. Skipping this video.")
        return
    frequency_param = parameter

    # 周波数を計算するための音声配列を作る作業    
    # 一秒間にfps*frequency_param個の音声データをサンプリングする
    audio_samples_frequency = np.array(audio.get_array_of_samples())
    # Resample audio to match desired frequency samples
    audio_samples_frequency = resample(audio_samples_frequency, int(total_frames * frequency_param)).astype(np.float32)

    # 音声データを必要な長さに調整する
    audio_samples = np.array(audio.get_array_of_samples())
    # resample audio to match video frame count
    audio_samples = resample(audio_samples, int(total_frames))

    if np.issubdtype(audio_samples.dtype, np.integer):
        audio_samples = audio_samples / np.iinfo(audio_samples.dtype).max
    elif np.issubdtype(audio_samples.dtype, np.floating):
        audio_samples = audio_samples / np.finfo(audio_samples.dtype).max

    process_bar = tqdm(total=total_frames)
    # 周波数情報を離散フーリエ変換で取得する
    results = []
    for i in range(int(total_frames)):
        start = i * frequency_param
        end = start + frequency_param

        S = librosa.feature.melspectrogram(y=audio_samples_frequency[start:end], sr=round(fps*frequency_param))
        plt.figure(figsize=(3,3))
        # データをdB単位に変換
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.imshow(S_dB, cmap='gray', origin='lower', aspect='auto')
        plt.axis('off')
        plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        img = Image.open(buf).convert('L').resize((frame_width, frame_height), Image.ANTIALIAS)
        img_array = np.array(img)
        results.append(img_array)
        plt.close()
        process_bar.update(1)
    print('result length: ',len(results))
    print('frequency length: ',len(audio_samples_frequency))
    print('audio length',len(audio_samples))
    print('resampled audio length', len(audio_samples))
    print('resampled audio frequency length', len(audio_samples_frequency))
    print('total frames',total_frames)
    # Iterate over each frame in the video
    frames = []
    pbar = tqdm(total=total_frames)
    frame_count=0

    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    if isfmrgb == 'false':
        output_file_path = os.path.join(output_dir, base_filename + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        ret_flow, frame_flow = flow_video.read()
        if not ret_flow:
            print("Could not read from flow output video. Stopping.")
            break

        # フレームをリサイズ
        frame_flow_resized = cv2.resize(frame_flow, (frame_width, frame_height))

        if not ret:
            break
        if frame_count < len(audio_samples):
            # if frame_count%skip == 0:
            print(frame_count)
            # RGBに変換する
            if isfmrgb=='true':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif isfmrgb=='false':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print('入力が適切ではありません')
                return
            # 正規化したスペクトログラムで画像を埋める
            spectrogram_channel = results[frame_count]

            if frame_count%skip == 0:
                frame = np.dstack((frame, frame_flow_resized[:,:,0], spectrogram_channel))
                # frames配列に追加
                frames.append(frame)
        else:
            print(f'Frame count {frame_count} exceeds audio sample length. Stopping processing.')
            break
        frame_count += 1
        pbar.update(1)
    
    frames = np.stack(frames)
    if isfmrgb=='true':
        output_file_path = os.path.join(output_dir, base_filename + '.pt')
        torch.save(torch.from_numpy(frames), output_file_path)
    if isfmrgb == 'false':
        for frame in frames:
            out_video.write(frame)
        out_video.release()
    video.release()
    flow_video.release()

# 動画を読み込み、音声を抽出し、フレームを処理してテンソルとして保存する関数
def process_video(video_path, max_frames, parameter, isfmrgb, skip, output_dir):
    # ビデオを読み込む
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ビデオのfpsを取得する
    fps = video.get(cv2.CAP_PROP_FPS)
    print('fps: ', fps)

    # flowoutput動画を読み込む
    flow_output_filename = os.path.splitext(os.path.basename(video_path))[0] + 'flowoutput.mp4'
    flow_output_path = os.path.join(os.path.dirname(video_path), flow_output_filename)
    flow_video = cv2.VideoCapture(flow_output_path)

    # ビデオのフレームの総数の情報を取得する
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    try:
        # mp4データから音声データを取り出す
        audio = AudioSegment.from_file(video_path, format="mp4") 
    except:
        print(f"Could not extract audio from {video_path}. Skipping this video.")
        return
    frequency_param = parameter

    # 周波数を計算するための音声配列を作る作業    
    # 一秒間にfps*frequency_param個の音声データをサンプリングする
    audio_samples_frequency = np.array(audio.get_array_of_samples())
    # Resample audio to match desired frequency samples
    audio_samples_frequency = resample(audio_samples_frequency, int(total_frames * frequency_param)).astype(np.float32)

    # 音声データを必要な長さに調整する
    audio_samples = np.array(audio.get_array_of_samples())
    # resample audio to match video frame count
    audio_samples = resample(audio_samples, int(total_frames))

    if np.issubdtype(audio_samples.dtype, np.integer):
        audio_samples = audio_samples / np.iinfo(audio_samples.dtype).max
    elif np.issubdtype(audio_samples.dtype, np.floating):
        audio_samples = audio_samples / np.finfo(audio_samples.dtype).max

    # 周波数情報を離散フーリエ変換で取得する
    results = []
    for i in range(int(total_frames)):
        start = i * frequency_param
        end = start + frequency_param

        frequencies, times, Zxx = signal.stft(y=audio_samples_frequency[start:end], fs=round(fps*frequency_param))
        mag = np.abs(Zxx)
        sum_mag = np.sum(mag, axis=-1)
        max_freq = frequencies[np.argmax(sum_mag)]
        results.append(max_freq)
    max_value = max(results)
    results = [(x/max_value)*255 for x in results]

    base_filename = os.path.splitext(os.path.basename(video_path))[0]

    # Iterate over each frame in the video
    frames = []
    pbar = tqdm(total=total_frames)
    frame_count = 0
    while True:
        ret, frame = video.read()
        ret_flow, frame_flow = flow_video.read()

        if not ret_flow:
            print("Could not read from flow output video. Stopping.")
            break

        # フレームをリサイズ
        frame_flow_resized = cv2.resize(frame_flow, (frame_width, frame_height))

        if not ret:
            break
        if frame_count < len(audio_samples):
            # RGBに変換する
            if isfmrgb=='true':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif isfmrgb=='false':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print('入力が適切ではありません')
                return
            # 正規化したスペクトログラムで画像を埋める
            spectrogram_channel = np.full(frame.shape[:2], int(results[frame_count]))
            if frame_count%skip == 0:
                # 5チャンネル画像を生成する
                frame = np.dstack((frame, frame_flow_resized[:,:,0], spectrogram_channel))
                # frames配列に追加
                frames.append(frame)
        else:
            print(f'Frame count {frame_count} exceeds audio sample length. Stopping processing.')
            break
        frame_count += 1
        pbar.update(1)

    # frames配列を構成する
    frames = np.stack(frames)

    # Concatenate frames and sound tensor along channel axis
    # Save as tensor
    output_file_path = os.path.join(output_dir, base_filename + '.pt')
    torch.save(torch.from_numpy(frames), output_file_path)

# 指定されたディレクトリ内の全ての動画を処理する関数 
def process_directory(opt):
    audio_process_method = opt.audiomethod
    directory_path = opt.target
    parameter = opt.frequency_param
    isfmrgb = opt.fmrgb
    skip = opt.skip
    output_dir = opt.output_dir
    max_frames = get_max_frames(directory_path)
    for filename in os.listdir(directory_path):
        if filename.endswith('.mp4'):
            if audio_process_method=='simple':
                process_video(os.path.join(directory_path, filename), max_frames, parameter, isfmrgb, skip, output_dir)
            elif audio_process_method=='mel':
                process_video_withmel(os.path.join(directory_path, filename), max_frames, parameter, isfmrgb, skip, output_dir)
            else:
                print('音声を処理するメソッドの指定が適切ではありません')


# main関数：コマンドライン引数を解析し、ディレクトリ内の動画の処理を開始する
if __name__=='__main__':
    start_time = datetime.datetime.now()
    print('start time:',start_time)
    sys.stdout.flush()
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=str, required=True, help='folder')
    parser.add_argument('--audiomethod',type=str, default='simple', help='audio method')
    parser.add_argument('--skip', type=int, default='1', help='num of skip')
    parser.add_argument('--frequency_param', type=int, default=100, help='frequency parameter')
    parser.add_argument('--fmrgb', type=str, default='true', help='fmrgb or not true false')
    parser.add_argument('--output_dir',type=str, required=True, help='output directory')
    opt = parser.parse_args()
    # 出力ディレクトリが存在するかどうかを確認
    if not os.path.exists(opt.output_dir):
        print(f"Output directory {opt.output_dir} does not exist. Creating it.")
        os.makedirs(opt.output_dir)
    print(opt)
    print('-----biginning processing-----')
    process_directory(opt)
    print('-----completing processing-----')
    sys.stdout.flush()
    # get end time of program
    end_time = datetime.datetime.now()
    # calculate execution time with start and end time
    execution_time = end_time - start_time
    print('end time:',end_time)
    sys.stdout.flush()
    print('Execution time: ', execution_time)
    sys.stdout.flush()