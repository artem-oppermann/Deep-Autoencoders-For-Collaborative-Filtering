# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:14:39 2018

@author: Artem Oppermann
"""

from model.base_model import BaseModel

class InferenceModel(BaseModel):
    
    def __init__(self, FLAGS):
        
        super(InferenceModel,self).__init__(FLAGS)
        self._init_parameters()