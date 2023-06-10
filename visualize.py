import cv2
import torch
import argparse
import numpy as np

def visualize(opt):
    data_name = opt.target
    result_path = opt.result
    # Load the tensor
    frames = torch.load(data_name).numpy()
    # Number of channels
    num_channels = frames.shape[-1]
    # Prepare a video writer for each channel
    outs = [cv2.VideoWriter(f'out_channel_{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frames.shape[2], frames.shape[1])) for i in range(num_channels)]


    # Iterate over each frame
    for frame in frames:
        # Convert each channel to grayscale and write to the video file
        for channel in range(frame.shape[-1]):
            gray_frame = cv2.cvtColor(np.uint8(frame[:, :, channel]), cv2.COLOR_GRAY2BGR)
            outs[channel].write(gray_frame)

    # Release the video writers
    for out in outs:
        out.release()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=str, required=True, help='pt data')
    parser.add_argument('--result',type=str, help='result_path')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning processing-----')
    visualize(opt)
    print('-----completing processing-----')