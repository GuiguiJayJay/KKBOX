"""
============================================
KKBox's Music Recommendation Challenge draft
============================================
Remember to enable ALL the 'prep_switch' to 'True' if you want to build the matching files.

"""
print(__doc__)

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import argparse
import os

import kklib.preproc as prep
import kklib.utils as utils
import kklib.models as mod
import kklib.settings as settings

# Initialize global variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
settings.init()
if settings.TIMER==True: time_start = time.time()

# Parse inputs
parser = argparse.ArgumentParser(description='Define important inputs.')
parser.add_argument("--datapath",
                    type=str,
                    default='data/',
                    help="Path to data directory")
                    
FLAGS, unparsed = parser.parse_known_args()


########################
# DATA PRE-PROCESSING
########################
# Dict. holding all data to be passed (either for training, either for testing, not both)
data = {'users': 0,
        'songs': 0,}
data[settings.trainingmode] = 0

# Dict. holding the matching data (either for training, either for testing, not both)
datamatch = {'songs': 0,
             'users': 0}
dataunmatch = {'songs': 0,
               'users': 0}
        
# Pre-processing and feat. eng.
prep.pp_all()

# Either features Binary Encoding, either One-Hot Encoding
for item in data.keys():              
  data[item] = prep.binary(key=item)

# Add the labels to data dict if training mode
if settings.trainingmode == 'train':
  filename = utils.namer(key='labels', suffix='_prep.csv')
  data['labels'] = np.fromfile(filename, dtype='uint8', count=-1, sep=',')


########################
# MACHINE LEARNING PART
########################
if settings.bypassAE == False:
  # Define and run (train or load) the Auto-Encoder for users dataset
  mod.AE_model(init_lr = 0.1,
               decay_rate = 0.95,
               batchsize = 1,
               epochs = 1,
               display_steps = 10000,
               num_hidden = 20,
               data = data,
               key = 'users')
  # Define and run (train or load) the Auto-Encoder for songs dataset
  mod.AE_model(init_lr = 0.1,
               decay_rate = 0.95,
               batchsize = 1,
               epochs = 1,
               display_steps = 100000,
               num_hidden = 50,
               data = data,
               key = 'songs')
  # Define and run (train or load) the Auto-Encoder for train dataset
  mod.AE_model(init_lr = 0.1,
               decay_rate = 0.95,
               batchsize = 1,
               epochs = 1,
               display_steps = 100000,
               num_hidden = 20,
               data = data,
               key = settings.trainingmode)

# Load the matching files
file_matched = utils.namer(key=settings.trainingmode, suffix='_matchedsongs.csv')
datamatch['songs'] = np.fromfile(file_matched, dtype='int32', count=-1, sep=',') 
file_matched = utils.namer(key=settings.trainingmode, suffix='_matchedusers.csv')
datamatch['users'] = np.fromfile(file_matched, dtype='int32', count=-1, sep=',')
# Load the unmatching files
file_matched = utils.namer(key=settings.trainingmode, suffix='_unmatchedsongs.csv')
dataunmatch['songs'] = np.fromfile(file_matched, dtype='int32', count=-1, sep=',') 
file_matched = utils.namer(key=settings.trainingmode, suffix='_unmatchedusers.csv')
dataunmatch['users'] = np.fromfile(file_matched, dtype='int32', count=-1, sep=',')

# Run the NN
mod.NN_model(init_lr = 0.01,
             decay_rate = 0.5,
             batchsize = 10,
             epochs = 3,
             display_steps = 10000,
             num_hidden1 = 500,
             num_hidden2 = 200,
             num_hidden3 = 100,
             data = data,
             matching = datamatch,
             unmatching = dataunmatch)

if settings.TIMER==True: print("--- Total running time = %0.2f s ---" % (time.time() - time_start))

########################
# TODO
########################
"""
- write predictions
- try using sigm after each layer
    
"""
