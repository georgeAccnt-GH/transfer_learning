### Copyright (C) Microsoft Corporation.  

import os
import numpy as np
import pandas as pd 

class trvis_consts(object):

    BASE_INPUT_DIR_list = ['data']
    PRETRAINED_MODELS_DIR_list = ['models', 'pretrained']
    PROCESSED_DATA_DIR_list = ['processed']

    def __setattr__(self, *_):
        raise TypeError


# os agnostic 'ls' function
def get_files_in_dir(crt_dir):
        return( [f for f in os.listdir(crt_dir) if os.path.isfile(os.path.join(crt_dir, f))])
