import pandas as pd
import numpy as np
import time

import kklib.settings as settings


############################################
def dataload(filename='data/train.csv'):
  """ load the data in a pandas dataframe.
  
  Parameters
  ==========
  filename: string
    location and name of the file to be read.
    
  Returns
  =======
  data: pandas.DataFrame
    the data in a pandas dataframe.
  """

  data = pd.read_csv(filepath_or_buffer=filename)
  # Note: if you read a numpy saved csv file, you need to set the option:
  #       header=None
  # else the first line of the data file will be skipped. Add an option for it later.

  return data


############################################
def namer(key='', suffix=None):
  """ Add a given suffix to an input file path, at a given separation point.
  
  Parameters
  ==========
  key: str
    String holding the python dict key of the dataset to be worked on.
    
  suffix: string
    Suffix to be appended.
    
  Returns
  =======
  outname: string
    The new file path.
  """
  
  filename = settings.path_dir + key + '.csv'
  if suffix == None:
    outname = filename
  else:
    outname = filename.split('.')
    outname = outname[0] + suffix

  return outname
  

############################################
def int2bin(label, maxlabel):
  """ Encode label of a given feature as binary encoded vector.
  
  Parameters
  ==========
  label: int
    The feature's label value to be binarized.
    
  maxlabel: int
    The feature's max array sizes necessary for binary encoding.
    
  Returns
  =======
  feature: np array
    the numpy array containing the binary-encoded value of a given feature.
  """

  feature = np.zeros(shape=maxlabel, dtype='uint8')
  counter = 0
  
  if label == 0:
      return feature
  
  while label:
      if label & 1 == 1: # mask of overlapping bits between the two numbers
          feature[counter] = 1
          counter += 1
      else:
          feature[counter] = 0
          counter += 1
      label = int(label/2)
        
  return feature


############################################
def sizefinder(maxval):
  """ Find the necessary array sizes depending on max possible value for a given feature.
  
  Parameters
  ==========
  maxval: int
    The feature's maximum label value across the data.
    
  Returns
  =======
  counter: int
    The needed array size to be used to encode the given feature as a binary vector.
  """
  
  counter = 0
  while maxval:
      counter += 1
      maxval = int(maxval/2)
      
  return counter
  

############################################
def maxfiller(key='users', featdict={}):
  """ Build a dict of required array sizes for binary encoding for each feature.
  
  Parameters
  ==========
  key: str
    String holding the python dict key of the dataset to be worked on.
    
  featdict: python dict
    Dict of feature names, and associated data.
    
  Returns
  =======
  None
  """
  
  if settings.MONITOR == True: print("feature sizes:")
  
  for feature in settings.maxcat[key].keys():
    if feature == 'timediff':
      settings.maxcat[key][feature] = 1
      settings.num_feat[key] += settings.maxcat[key][feature]
      if settings.MONITOR == True: print("%s \t %d" % (feature, settings.maxcat[key][feature]) )
    else:
      settings.maxcat[key][feature] = sizefinder(max(featdict[feature]))
      settings.num_feat[key] += settings.maxcat[key][feature]
      if settings.MONITOR == True: print("%s \t %d" % (feature, settings.maxcat[key][feature]) )

  if settings.MONITOR == True: print("Tot num. of features: %s" % settings.num_feat[key])    
                              
  return None


############################################
def data2mat(filename='data/users.csv', key='users'):
  """ Conversion to dict of numpy arrays of preprocessed data.
  
  Parameters
  ==========
  filename: string
    location and name of the file to be read.
    
  key: str
    String holding the python dict key of the dataset to be converted.
    
  Returns
  =======
  outdict: python dict
    A dictionnary holding the feature names, and a corresponding numpy array 
    to hold their data after label-encoding.
  """
  
  if settings.TIMER==True: time_start = time.time()
  
  # Load the pre-proc'd dataset
  data = dataload(filename=filename)

  outdict = {}
  
  # Numpy array conversion
  for feat in settings.maxcat[key].keys():
    outdict[feat] = data[feat].as_matrix()
    # Number of data entries
    if settings.num_data[key] == 0:
      settings.num_data[key] = len(outdict[feat])

  if settings.TIMER==True: print("-> Matrix conv. = %0.2f s" % (time.time() - time_start))

  return outdict
  
  
############################################
def batch_AE(step=0, batchsize=1, data={}, key='train'):
  """ Creates a batch from single pre-processed data file.
  
  Parameters
  ==========
  step: int
    Step in the current training/test process.
    
  batchsize: int
    Size of the batch.
    
  data: python dict
    Dictionnary holding data names and associated data (either sklearn sparse matrix
    if OHE, or np.array if BE)
    
  key: str
    Indicate if one should build a batch for training or testing (thus possible values 
    are 'train' and 'test' only).

  Returns
  =======   
  batch: np.array
    The batch of dataq.
  """
  
  # Absolute begining and end of current batch in data
  start = step * batchsize
  stop = start + batchsize
  
  # adapt size of the last batch
  if (start + batchsize) > settings.num_data[key]:
      batchsize = settings.num_data[key] - start
  
  batch = data[key][start:stop]
  
  return batch


############################################
def batch_full(step=0,
               batchsize=1,
               data={},
               matching={},
               unmatching={}, 
               order=None):
  """ Creates a batch from all the data files.
  
  Parameters
  ==========
  step: int
    Step in the current training/test process.
    
  batchsize: int
    Size of the batch.
    
  data: python dict
    Dictionnary holding data names and associated data (either sklearn sparse matrix
    if OHE, or np.array if BE)
    
  matching: python dict
    Dictionnary holding datanames ('train' or 'test') and the associated python dict
    which holds 2 matching arrays and their names ('songs' and 'users').

  unmatching: python dict
    Dictionnary holding datanames ('train' or 'test') and the associated data entries
    for which no data where found in songs/users.
    
  order: np.array
    An array of random permutations to read data.

  Returns
  =======   
  batch: np.array
    The batch of datas.
    
  target: np.array
    The labels associated to the previously built batch.
  """
  
  # Absolute begining and end of current batch in data
  start = step * batchsize
  stop = start + batchsize
  
  # adapt size of the last batch
  if stop > settings.num_data[settings.trainingmode]:
      stop = settings.num_data[settings.trainingmode]
      batchsize = stop - start
   
  tempdata = []
  templist = {'users': [], 'songs': []}
  tempmat = {'users': 0, 'songs': 0}
  dummy1 = np.zeros(shape=[1, settings.num_feat['users']], dtype='float32')
  dummy2 = np.zeros(shape=[1, settings.num_feat['songs']], dtype='float32')
  dummy = {'users': dummy1, 'songs': dummy2}

  if settings.bypassAE == False:
    # Finding the good songs/users data and concatenate them into a single matrix
    for item in templist.keys():
      # Go through each train/test data entry of the batch to find matching users/songs data
      for i in range(start,stop):
        # Apply the NN weights to the matching data, then append it to the data list
        if order[i] not in unmatching[item]:
          templist[item].append(np.matmul(data[item][matching[item][order[i]]], settings.NNweightsdict[item]))
        # Place dummy values for unmatched songs
        else:
          templist[item].append(np.matmul(dummy[item], settings.NNweightsdict[item]))
          
      # Concatenate all the data entries into a single matrix, and append it to the temp list
      tempmat[item] = np.vstack(templist[item])
      tempdata.append(tempmat[item])
    
    # Deal with the Training/Testing set
    # Apply the NN weights to the train/test data, then append it to the data list
    tempdata.append(np.matmul(data[settings.trainingmode][order[start:stop]],
                                   settings.NNweightsdict[settings.trainingmode]))
      
  else:
    # Finding the good songs/users data and concatenate them into a single matrix
    for item in templist.keys():
      # Go through each train/test data entry of the batch to find matching users/songs data
      for i in range(start,stop):
        if np.any(order[i] == unmatching[item]):
          templist[item].append(dummy[item])
        # Place dummy values for unmatched songs
        else:
          templist[item].append(data[item][matching[item][order[i]]])
          
      # Concatenate all the data entries into a single matrix, and append it to the temp list
      tempmat[item] = np.vstack(templist[item])
      tempdata.append(tempmat[item])
    
    # Deal with the Training/Testing set
    tempdata.append(data[settings.trainingmode][order[start:stop]])
    
  # Concatenate all the 3 matrices together
  batch = np.hstack(tempdata)
  
  if settings.trainingmode == 'train':
    # Get the corresponding labels
    target = data['labels'][order[start:stop]]
    target = np.reshape(target, [batchsize,1])
  else:
    target = None
  
  return batch, target
  
