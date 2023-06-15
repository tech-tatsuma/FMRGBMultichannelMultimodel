import os
import argparse

def rename(opt):
    folder_path = opt.target

    # ディレクトリ内のすべてのファイルをリストアップします
    for filename in os.listdir(folder_path):
        # ファイル名が'resized_'で始まる場合、そのプレフィックスを削除します
        if filename.startswith('resized_'):
            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, filename.replace('resized_', '', 1))
            
            # 新しい名前が既に存在しないことを確認します
            if not os.path.exists(dst):
                os.rename(src, dst)
                print(f'ファイル名を{filename}から{dst}に変更しました。')
            else:
                print(f'ファイル名{dst}は既に存在します。')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=str, required=True, help='target')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning processing-----')
    rename(opt)
    print('-----completing processing-----')
