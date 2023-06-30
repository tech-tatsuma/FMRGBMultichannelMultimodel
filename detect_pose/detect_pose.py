import os
import subprocess
import argparse

def detect_pose(opt):
    # ビデオファイルのディレクトリ
    video_dir = opt.video_path

    # ディレクトリ内のすべてのファイルを処理
    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):  # mp4ファイルのみを処理
            video_path = os.path.join(video_dir, filename)
            
            subprocess.run(['ffmpeg', '-i', video_path, '-vf', 'scale=852:ceil(ow/a/2)*2', video_path])
            # 外部のdetect.pyスクリプトを呼び出す
            subprocess.run(['python', 'demo_video.py', video_path])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path',type=str, required=True, help='video path')
    opt = parser.parse_args()
    print(opt)
    detect_pose(opt)