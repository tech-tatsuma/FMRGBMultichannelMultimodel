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

def stereo_to_mono(audio_samples):
    return audio_samples[::2] / 2 + audio_samples[1::2] / 2

# 動画を読み込み、音声を抽出し、フレームを処理してテンソルとして保存する関数
def process_video_withmel(video_path, max_frames):
    # Load video
    video = cv2.VideoCapture(video_path)

    # Extract audio
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio_samples = np.array(audio.get_array_of_samples())

    if audio.channels == 2:
        audio_samples = stereo_to_mono(audio_samples)

    if np.issubdtype(audio_samples.dtype, np.integer):
        audio_samples = audio_samples / np.iinfo(audio_samples.dtype).max
    elif np.issubdtype(audio_samples.dtype, np.floating):
        audio_samples = audio_samples / np.finfo(audio_samples.dtype).max

    # Generate spectrogram
    spectrogram = librosa.feature.melspectrogram(audio_samples, n_mels=128)
    db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Resize spectrogram to match video frames
    db_spectrogram_resized = zoom(db_spectrogram, [max_frames / db_spectrogram.shape[1], 1], mode='nearest')
    db_spectrogram_resized = db_spectrogram_resized.T # Transpose to match video frames
    db_spectrogram_resized = (db_spectrogram_resized - db_spectrogram_resized.min()) / (db_spectrogram_resized.max() - db_spectrogram_resized.min()) * 255

    # Iterate over each frame in the video
    frames = []
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Additional channels for frequency and amplitude
        f, t, Sxx = signal.spectrogram(audio_samples)
        max_freq = f[np.argmax(Sxx)]
        max_amp = np.max(Sxx)
        freq_channel = np.full(frame.shape, max_freq)
        amp_channel = np.full(frame.shape, max_amp)

        # Stack the channels to the frame
        frame = np.dstack((frame, freq_channel, amp_channel))
        frames.append(frame)

    frames = np.stack(frames)

    # Concatenate frames and sound tensor along channel axis
    # Save as tensor
    tensor = {'frames': torch.from_numpy(frames)}
    torch.save(tensor, video_path + '.pt')

# 動画を読み込み、音声を抽出し、フレームを処理してテンソルとして保存する関数
def process_video(video_path, max_frames):
    # Load video
    video = cv2.VideoCapture(video_path)

    # Extract audio
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio_samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        audio_samples = stereo_to_mono(audio_samples)
    if np.issubdtype(audio_samples.dtype, np.integer):
        audio_samples = audio_samples / np.iinfo(audio_samples.dtype).max
    elif np.issubdtype(audio_samples.dtype, np.floating):
        audio_samples = audio_samples / np.finfo(audio_samples.dtype).max

    # Generate spectrogram
    D = librosa.stft(audio_samples)
    spectrogram = np.abs(D)
    db_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Resize spectrogram to match video frames
    db_spectrogram_resized = zoom(db_spectrogram, [max_frames / db_spectrogram.shape[1], 1], mode='nearest')
    db_spectrogram_resized = db_spectrogram_resized.T # Transpose to match video frames
    db_spectrogram_resized = (db_spectrogram_resized - db_spectrogram_resized.min()) / (db_spectrogram_resized.max() - db_spectrogram_resized.min()) * 255

    # Iterate over each frame in the video
    frames = []
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Additional channels for frequency and amplitude
        f, t, Sxx = signal.spectrogram(audio_samples)
        max_freq = f[np.argmax(Sxx)]
        max_amp = np.max(Sxx)
        freq_channel = np.full(frame.shape, max_freq)
        amp_channel = np.full(frame.shape, max_amp)

        # Stack the channels to the frame
        frame = np.dstack((frame, freq_channel, amp_channel))
        frames.append(frame)

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
