# -*- coding: utf-8 -*-
##########################################################################
# Created on Tue Jun 25 13:25:41 2013
# Copyright (c) 2013-2021, CEA/DRF/Joliot/NeuroSpin. All rights reserved.
# @author:  Edouard Duchesnay
# @email:   edouard.duchesnay@cea.fr
# @license: BSD 3-clause.
##########################################################################


# System import
from setuptools import setup, find_packages
import os


release_info = {}
infopath = os.path.join(os.path.dirname(__file__), "mulm", "info.py")
with open(infopath) as open_file:
    exec(open_file.read(), release_info)
pkgdata = {
    "mulm": ["tests/*.py", ],
}

setup(
    name=release_info["NAME"],
    description=release_info["DESCRIPTION"],
    long_description=release_info["LONG_DESCRIPTION"],
    license=release_info["LICENSE"],
    classifiers=release_info["CLASSIFIERS"],
    author=release_info["AUTHOR"],
    author_email=release_info["AUTHOR_EMAIL"],
    version=release_info["VERSION"],
    url=release_info["URL"],
    packages=find_packages(exclude="doc"),
    platforms=release_info["PLATFORMS"],
    extras_require=release_info["EXTRA_REQUIRES"],
    install_requires=release_info["REQUIRES"],
    package_data=pkgdata,
    scripts=release_info["SCRIPTS"]
)
