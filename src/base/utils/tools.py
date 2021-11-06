import collections
from contextlib import contextmanager
import os
import subprocess
import time
import numpy as np
from matplotlib import pyplot as plt


class Tools():
    @staticmethod
    def normalize(value, mean, std):
        return (value - mean) / std

    @staticmethod
    def inv_trans(value, mean, std):
        return value * std + mean

    @staticmethod
    def txtread(path):
        lines = []
        with open(path) as f:
            for l in f: lines.append(l)
        return lines

    @staticmethod
    def imread(path, size=None, gray=False):
        import cv2
        img = cv2.imread(str(path))
        if img is None: print(f"##### can't load {path} #####")
        img = img[:,:,::-1]
        if size is not None:
            img = Tools.resize(img, size)
        if gray:
            img = img[:,:,0]
        return img

    @staticmethod
    def resize(img, size, nearest=False, area=False):
        import cv2
        if nearest:
            mode = cv2.INTER_NEAREST
        elif area:
            mode = cv2.INTER_AREA
        else:
            mode = cv2.INTER_CUBIC
        size = (size[1], size[0])
        img = cv2.resize(img, size, interpolation=mode)
        return img

    @staticmethod
    def imshow(img, figsize=None, title=None):
        if figsize is not None:
            plt.figure(figsize=figsize)
        if title is not None:
            plt.title(title)
        plt.imshow(img)
        plt.show()

    @staticmethod
    def imsave(img, path):
        import cv2
        if len(img.shape) == 3:
            img_s = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else: img_s = img
        cv2.imwrite(str(path), img_s)

    @staticmethod
    @contextmanager
    def timer(name):
        t0 = time.time()
        yield
        print(f'[{name}] done in {time.time() - t0:.8f} s')

    @staticmethod
    def flatten(l):
        for el in l:
            if isinstance(el, collections.abc.Iterable) and \
               not isinstance(el, (str, bytes)):
                yield from Tools.flatten(el)
            else:
                yield el

    @staticmethod
    def make_dir(path, ignore=True):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if not ignore:
                print('{} already exists.'.format(path))

    @staticmethod
    def get_git_hash():
        cmd = "git rev-parse --short HEAD"
        _hash = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode('utf-8')
        return _hash

    @staticmethod
    def get_git_diff():
        cmd = "git diff"
        diff = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode('utf-8')
        diff = '\n'.join(['\t' + l for l in diff.split('\n')])
        return diff

    @staticmethod
    def npread(path, size=None):
        img = np.load(str(path))
        if size is not None:
            img = Tools.resize(img, size)
        return img

    @staticmethod
    def scale(x):
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def dict_merge(dct, merge_dct):
        for k, v in merge_dct.items():
            if (k in dct and isinstance(dct[k], dict)
                    and isinstance(merge_dct[k], collections.Mapping)):
                Tools.dict_merge(dct[k], merge_dct[k])
            else:
                dct[k] = merge_dct[k]
        return dct
