import os
import yaml

def get_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def save_to_yaml(files, output_file):
    data = {'files': files}
    with open(output_file, 'w') as f:
        yaml.dump(data, f)

if __name__ == '__main__':
    current_directory = os.getcwd()
    files = get_files(current_directory)
    output_file = 'qlib.yaml'
    save_to_yaml(files, output_file)
    print(f"文件名已保存到 {output_file} 中。")
