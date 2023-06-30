import os
import argparse

def delete_not_resized(opt):
    # 指定されたディレクトリの中のファイルをチェックします
    folder_path = opt.target

    # ディレクトリ内のすべてのファイルをリストアップします
    for filename in os.listdir(folder_path):
        # ファイル名が'resized'で始まらない場合、そのファイルを削除します
        if not filename.startswith('resized'):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f'ファイル{filename}を削除しました。')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help='target')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning processing-----')
    delete_not_resized(opt)
    print('-----completing processing-----')