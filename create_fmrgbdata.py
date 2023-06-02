import os
import cv2
import numpy as np
import torch
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import zoom
import argparse

def get_max_frames(directory_path):
    max_frames = float('-inf')
    for filename in os.listdir(directory_path):
        if filename.endswith('.mp4'):
            video = cv2.VideoCapture(os.path.join(directory_path, filename))
            frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if frames > max_frames:
                max_frames = frames
    return max_frames

def process_video(video_path, max_frames):
    # Load video
    video = cv2.VideoCapture(video_path)

    # Extract audio
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio.export("out.wav", format="wav")
    sample_rate, samples = wavfile.read('out.wav')

    # Get frequencies and amplitudes
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

def process_directory(opt):
    directory_path = opt.target
    max_frames = get_max_frames(directory_path)
    for filename in os.listdir(directory_path):
        if filename.endswith('.mp4'):
            process_video(os.path.join(directory_path, filename), max_frames)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=str, required=True, help='folder')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning processing-----')
    process_directory(opt)
    print('-----completing processing-----')
