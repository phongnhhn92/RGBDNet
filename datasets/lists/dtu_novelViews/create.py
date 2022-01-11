from os import listdir


file = open("list.txt", "w")
path = '/media/phong/Data2TB/dataset/DTU/mvs_training/dtu/Rectified'
scans = listdir(path)

for scan in scans:
    file.writelines(scan[:-6]+'\n')
file.close()