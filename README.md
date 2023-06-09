# FMRGBMultichannelMultimodel
The model is trained by creating video data with sound frequencies and amplitudes and image RGB information as channels.
The network uses a three-dimensional convolutional neural network.

## Usage
1. Creation of data sets
```
python create_fmrgbdata.py --target <mp4データが格納されているディレクトリーパス> --audiomethod <simple or mel> --frequency_param <周波数解析パラメータ（int）> --skip <ループをスキップする回数（int）>
```
target: Specification of the directory containing the video data to be converted to fmrgb data.

audiomethod: Option to analyse speech data with a simple spectrogram or with a melospectrogram.

frequency_param: Parameters used for frequency analysis. Specified as integer type.
    
2. Data visualisation
```
python visualize.py --target <ptファイル>
```
3. FMRGB multi-channel model training.
```
python train.py --data <csvデータ> --epochs <学習回数> --test_size <テストデータの割合指定> --patience <早期終了パラメータ>
```
##  Explanation of the program
### Imaging of audio data
SIMPLE SPECTROGRAM

1. Calculation of the STFT (Short Time Fourier Transform)：The librosa.stft(audio_samples) function is used to perform a short-time Fourier transform. It divides the audio signal into small chunks (frames) and applies the Fourier transform to each frame to produce a complex result.

2. Calculation of spectrograms：np.abs() is used to calculate an amplitude spectrogram from the STFT results. This shows the amplitude of the speech signal at each frame and frequency.

3. The frequencies with the highest amplitude (most affected by the sound) are taken from the results calculated in 2.

4. The results calculated by the above methods are scaled to 0~255 and embedded in the image.

MELSPECTROGRAM

1. Samples audio data at a constant frequency based on the video frame rate.

2. Use librosa.feature.melspectrogram to compute the melspectrogram of the speech data.

3. Save the mel spectrogram as an image and save its value as a NumPy array as the fifth channel image.

### How to align data lengths
1. Examines all videos in the directory and retrieves the frame count of the video with the highest frame count.

2. The number of frames is then resized to the maximum number of frames if the number of frames in the video is less than the maximum number of frames when each video is processed. The scipy.ndimage.zoom function is used for resizing. This function scales up or down each dimension of the input array based on a specific zoom factor. The zoom factor is calculated by dividing the current number of frames by the maximum number of frames.

### Processing by motion capture
https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0　

Access the URLs body_pose_model.pth and hand_pose_model.pth and download them. After the download is complete, move the pth files to detect_pose/model/.

Afterwards, go to the detect_pose directory and execute the following command. By executing this command, motion capture (openpose) is performed on all videos in the specified directory.
```
pip install -r requirements.txt
python detect_pose.py --video_path <動画データが入っているディレクトリのパス>
```
