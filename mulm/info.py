# -*- coding: utf-8 -*-
##########################################################################
# CEA - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Module current version
version_major = 0
version_minor = 0
version_micro = 1

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)


# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
Massive univariate linear model.
"""
SUMMARY = """
.. container:: summary-carousel

    Provide basic features similar to "statmodels" (like OLS) where Y is a
    matrix of many responses where many independant fit are requested.
"""
long_description = (
    "Massive univariate linear model.\n")

# Main setup parameters
NAME = "pylearn-mulm"
ORGANISATION = "CEA"
MAINTAINER = "Edouard Duchesnay"
MAINTAINER_EMAIL = "edouard.duchesnay@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "NeuroSpin"
EXTRAURL = "https://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/NeuroSpin.aspx"
URL = "https://github.com/neurospin/pylearn-mulm"
DOWNLOAD_URL = "https://github.com/neurospin/pylearn-mulm"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = """
pylearn-mulm developers
"""
AUTHOR_EMAIL = "edouard.duchesnay@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["mulm"]
REQUIRES = [
    "numpy>=1.17.1",
    "scipy>=0.19.1"
]
EXTRA_REQUIRES = {
}
SCRIPTS = [
]
