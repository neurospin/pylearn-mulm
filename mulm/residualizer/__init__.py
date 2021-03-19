# -*- coding: utf-8 -*-
##########################################################################
# Created on Tue Jun 25 13:25:41 2013
# Copyright (c) 2013-2021, CEA/DRF/Joliot/NeuroSpin. All rights reserved.
# @author:  Edouard Duchesnay
# @email:   edouard.duchesnay@cea.fr
# @license: BSD 3-clause.
##########################################################################

"""
Module that contains the residualizers.
"""

from .residualizer import Residualizer
from .residualizer import ResidualizerEstimator


__all__ = ['Residualizer',
           'ResidualizerEstimator']
