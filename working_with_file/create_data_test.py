import shutil
import os

origin_images_path = "/home/lmtruong1512/Pictures/Data/animals10/raw-img"
generate_images_path = "/home/lmtruong1512/Pictures/Data/animals10/img-test"
try:
    os.mkdir(generate_images_path)
except Exception as a:
    print("Error creating directory")
list_dir = os.listdir(origin_images_path)
numfolder = 0
for dir_name in list_dir:
    ori_full_path_dir = os.path.join(origin_images_path, dir_name)
    gen_full_path_dir = os.path.join(generate_images_path, dir_name)
    os.mkdir(gen_full_path_dir)
    list_files = os.listdir(ori_full_path_dir)
    numfolder += 1
    print("folder:", numfolder)
    numfile = 0
    for file_name in list_files:
        origin_file = os.path.join(ori_full_path_dir, file_name)
        generate_file = os.path.join(gen_full_path_dir, file_name)
        shutil.copyfile(origin_file, generate_file)
        numfile += 1
        print("file:", numfile)
        if numfile >= 100:
            break


