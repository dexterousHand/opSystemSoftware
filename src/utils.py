import os
import os.path as op
import glob
import cv2
import shutil
import numpy as np
from IPython import embed
from datetime import datetime

def ensure_dir(path, renew=False):
    """ Ensure the directory exists. Delete the old one if renew """
    if op.exists(path):
        if renew:
            shutil.rmtree(path)
        else:
            return
    os.makedirs(path)

def info(s, domain=""):
    """ Highlight info """
    domain_str = "" if len(domain) < 1 else ("".join(list(map(lambda x: "[{:^8}]".format(
        str(x)), domain if isinstance(domain, list) else [domain]))))
    with open("log.txt", 'a') as f:
        f.write("[INFO]" + "[{}]".format(str(datetime.now())) + domain_str + s + '\n')
    print("\033[96m{} \033[00m {}" .format("[INFO]" + domain_str, s))

"""
def bar(current, total, prefix="", suffix="", bar_sz=25, end_string=None):
    sp = ""
    print("\x1b[2K\r", end='')
    for i in range(bar_sz):
        if current * bar_sz // total > i:
            sp += '='
        elif current * bar_sz // total == i:
            sp += '>'
        else:
            sp += ' '
    if current == total:
        if end_string is None:
            print("\r%s[%s]%s" % (prefix, sp, suffix))
        else:
            if end_string != "":
                print("\r%s" % end_string)
            else:
                print("\r", end='')
    else:
        print("\r%s[%s]%s" % (prefix, sp, suffix), end='')
"""