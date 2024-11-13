import os
import shutil

def Delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f'folder {path} is removed')
    else:
        print(f'folder {path} not exist')

def Delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f'file {path} is removed')
    else:
        print(f'file {path} not exist')

def List_dir(path):
    return sorted(os.listdir(path))

def add_path_and_file(sub_dir,file_list):
    file = []
    for i in range(len(file_list)):
        file.append(os.path.join(sub_dir,file_list[i]))
    return file

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(f'{path}')
        print(f'dir {path} is made')
    else:
        print(f'dir {path} is exist')


def remove_and_make(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f'{path} is removed')
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'{path} is made')