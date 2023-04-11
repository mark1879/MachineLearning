#! /usr/bin/python
# encoding: utf-8

import os
import sys

cur_dir = os.getcwd()
pkg_rootdir = os.path.dirname(cur_dir)

# print(pkg_rootdir)

if pkg_rootdir not in sys.path:
    sys.path.append(pkg_rootdir)

# print(sys.path)