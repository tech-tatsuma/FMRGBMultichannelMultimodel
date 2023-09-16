# FMRGBMultichannelMultimodel(Multi task Regression Problem Version)
This program is designed for machine learning of data comprising sound, optical flow and visual information.

## Usage
### resize mp4 data
```
python resize_video.py --target <target folder> --mag <degree of resolution reduction>
```
When the mag is 3, the vertical and horizontal resolution is 1/3 each.

### Creation of data sets
```
python create_fmrgbdata.py --target <target folder> --audiomethod <simple or mel> --frequency_param <Frequency analysis parameters（int）> --skip <num of skip（int）> --isfmrgb <3 or 5 channel?true or false>
```
If isfmrgb is false, processing is performed on directories created by robustvideomatting.py in the following repositories.
https://github.com/kyutech-tatsuma/robustvideomatting.git
    
### Data visualisation
```
python visualize.py --target <pt>
```
### FMRGB multi-channel model training.
```
python train.py --data <csv> --epochs <epochs> --train_size <rate of traindata> --patience <early stopping parameter> --lr <learning rate> --rankloss <use ranking loss?> --learnmethod <conv3d convlstm vivit> --islearnrate_search <search learning rate?> --usescheduler <use lr scheduler?> --seed <value of seed>
```
##  Explanation of the program
### Imaging of audio data
<strong>SIMPLE SPECTROGRAM</strong>

1. Calculation of the STFT (Short Time Fourier Transform)：The librosa.stft(audio_samples) function is used to perform a short-time Fourier transform. It divides the audio signal into small chunks (frames) and applies the Fourier transform to each frame to produce a complex result.

2. Calculation of spectrograms：np.abs() is used to calculate an amplitude spectrogram from the STFT results. This shows the amplitude of the audio signal at each frame and frequency.

3. The frequencies with the highest amplitude are taken from the results calculated in 2.

4. The results calculated by the above methods are scaled to 0~255 and embedded in the image.

<strong>MELSPECTROGRAM</strong>

1. Samples audio data at a constant frequency based on the video frame rate.

2. Use librosa.feature.melspectrogram to compute the melspectrogram of the speech data.

3. Save the melspectrogram as an image and save its value as a NumPy array as the fifth channel image.

### How to align data lengths
1. Examines all videos in the directory and retrieves the frame count of the video with the highest frame count.

2. The number of frames is then resized to the maximum number of frames if the number of frames in the video is less than the maximum number of frames when each video is processed. The scipy.ndimage.zoom function is used for resizing. This function scales up or down each dimension of the input array based on a specific zoom factor. The zoom factor is calculated by dividing the current number of frames by the maximum number of frames.

## Install required module
```
apt-get update
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0
apt-get install ffmpeg
```
