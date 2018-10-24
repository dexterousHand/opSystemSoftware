#!/usr/bin/env python
# encoding: utf-8

import os
import shutil
import os.path as op
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

