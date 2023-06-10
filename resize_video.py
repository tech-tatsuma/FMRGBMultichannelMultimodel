import os
import subprocess
import argparse
from tqdm import tqdm

def resize_videos(opt):
    directory = opt.target
    # 指定されたディレクトリ内のファイルを取得
    file_list = os.listdir(directory)
    mag = opt.mag
    # mp4ファイルのみを選択
    video_files = [file for file in file_list if file.endswith(".mp4")]

    pbar = tqdm(total=len(video_files))

    for file in video_files:
        input_path = os.path.join(directory, file)
        output_path = os.path.join(directory, f"resized_{file}")
        
        # FFmpegコマンドを生成
        ffmpeg_cmd = f"ffmpeg -i {input_path} -vf scale=iw/{mag}:ih/{mag} {output_path}"
        
        # FFmpegコマンドを実行
        subprocess.call(ffmpeg_cmd, shell=True)
        
        print(f"{file}の解像度を変更しました。")
        pbar.update(1)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=str, required=True, help='folder')
    parser.add_argument('--mag',type=int, required=True, help='mag')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning processing-----')
    resize_videos(opt)
    print('-----completing processing-----')