import os
import glob
import argparse
def delete_mp4_files(opt):
    directory = opt.target
    # 指定されたディレクトリ内のすべての.mp4ファイルに対して
    for filename in glob.glob(os.path.join(directory, '*.mp4')):
        # ファイルを削除する
        os.remove(filename)
        print(f"Deleted file : {filename}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=str, required=True, help='target')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning processing-----')
    delete_mp4_files(opt)
    print('-----completing processing-----')


