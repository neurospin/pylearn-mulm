# -*- coding: utf-8 -*-
##########################################################################
# Created on Created on Thu Feb  6 15:15:14 2020
# Copyright (c) 2013-2021, CEA/DRF/Joliot/NeuroSpin. All rights reserved.
# @author:  Edouard Duchesnay
# @email:   edouard.duchesnay@cea.fr
# @license: BSD 3-clause.
##########################################################################

"""
The Massive Univariate Linear Model (MULM) module.
"""

from .models import MUPairwiseCorr
from .models import MUOLS
from .helpers import design_matrix

__all__ = ['MUPairwiseCorr',
           'MUOLS',
           'design_matrix']
