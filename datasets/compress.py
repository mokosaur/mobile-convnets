import cv2
import tarfile
import os
import glob
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--compress', help='sum the integers (default: find the max)')
parser.add_argument('-d', '--decompress', help='sum the integers (default: find the max)')
args = parser.parse_args()

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def compress(dir_name):
    tmpdir_name = downsample(dir_name)
    tar = tarfile.open(dir_name + ".tar.gz", "w:gz")
    for file_name in glob.glob(os.path.join(tmpdir_name, "*")):
        tar.add(file_name, os.path.basename(file_name))
    tar.close()
    shutil.rmtree(tmpdir_name)


def decompress(dir_name, archive_name=None):
    if not archive_name:
        archive_name = dir_name
    shutil.rmtree(dir_name)
    create_dir(dir_name)
    tar = tarfile.open(archive_name + ".tar.gz", "r:gz")
    for tarinfo in tar:
        tar.extract(tarinfo, dir_name)
    tar.close()


def downsample(dir_name, dest_size=448):
    tmpdir_name = dir_name + '-tmp'
    create_dir(tmpdir_name)
    for class_name in os.listdir(dir_name):
        create_dir(os.path.join(tmpdir_name, class_name))
        for file_name in os.listdir(os.path.join(dir_name, class_name)):
            im = cv2.imread(os.path.join(dir_name, class_name, file_name), 1)
            height, width, _ = im.shape
            shorter_edge = min(height, width)
            if shorter_edge > dest_size:
                ratio = dest_size / shorter_edge
                im = cv2.resize(im, (int(height * ratio), int(width * ratio)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(tmpdir_name, class_name, file_name.split('.')[0] + '.jpg'), im)
    return tmpdir_name


if __name__ == "__main__":
    if args.compress:
        print("Compressing files...")
        compress(os.path.join(__location__, args.compress))
    if args.decompress:
        print("Decompressing files...")
        decompress(os.path.join(__location__, args.decompress))
