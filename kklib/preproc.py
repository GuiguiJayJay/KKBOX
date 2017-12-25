import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

import kklib.utils as utils
import kklib.settings as settings



############################################
def pp_songs(filename='data/songs.csv', features={}):
  """ feat eng. and cleaning of the 'songs' datafile. Rewrite the result 
  as a new csv file for further processing.
  
  Parameters
  ==========
  filename: string
    location and name of the file to be read.
    
  features: python dict
    A dictionnary holding the feature name and the needed array sizes for binary encoding.
    The latter is not used here, we only need the feature names from the dict.
    
  Returns
  =======
  None
  """

  songs = utils.dataload(filename=filename)
  le = LabelEncoder()
  
  # composer or lyricist NaN = artist_name
  songs.loc[songs['composer'].isnull(), 'composer'] = songs['artist_name']
  songs.loc[songs['lyricist'].isnull(), 'lyricist'] = songs['artist_name']

  # split songs length into 3 categories
  songs.loc[songs['song_length']<200000, 'song_length'] = 0
  songs.loc[(songs['song_length']<250000) & (songs['song_length']>=200000), 'song_length'] = 1
  songs.loc[songs['song_length']>=250000, 'song_length'] = 2

  # Manually rewrite one entry badly written
  songs.loc[songs['language'].isnull(), 'lyricist'] = songs['artist_name']
  songs.loc[songs['language'].isnull(), 'language'] = 31

  # remove lowly populated genre_ids
  var_dict = songs['genre_ids'].value_counts().to_dict() # dict of {genre: counts}
  limit = 10
  misc =  {k: v for k, v in var_dict.items() if v <= limit} # genre to be merged
  for genres in misc.keys():
      songs.loc[songs['genre_ids']==genres, 'genre_ids'] = 'Misc'
  del misc, var_dict # clear temp obj
  songs['genre_ids'] = songs['genre_ids'].astype(str) # uniformization to string

  # Switch "Various Artists" in "artist_name" by something more relevant when possible
  songs.loc[(songs['artist_name']=='Various Artists') & (songs['composer']!='Various Artists'),
             'artist_name'] = songs['composer']
  songs.loc[(songs['artist_name']=='Various Artists') & (songs['lyricist']!='Various Artists'),
             'artist_name'] = songs['lyricist']

  # Label encoding and  32-bits unsigned int conversion
  for feat in features.keys():
    songs[feat] = le.fit_transform(songs[feat]) 
    songs[feat] = songs[feat].astype(np.uint32)
    
  # Write preprocessed file to disk
  filename = utils.namer(key='songs', suffix='_prep.csv')
  songs.to_csv(path_or_buf=filename, index=False)
  
  print("songs_prep file written")

  return None


############################################
def pp_users(filename='data/members.csv', features={}):
  """ feat eng. and cleaning of the 'members' datafile. Rewrite the result 
  as a new csv file for further processing.
  
  Parameters
  ==========
  filename: string
    location and name of the file to be read.

  features: python dict
    A dictionnary holding the feature name and the needed array sizes for binary encoding.
    The latter is not used here, we only need the feature names from the dict.
    
  Returns
  =======
  None
  """

  users = utils.dataload(filename=filename)
  le = LabelEncoder()
  
  # create an additional category for unknown gender
  users['gender'].fillna('it', inplace=True)

  # move lowest pop. cat. to most similar cat.
  users.loc[(users['registered_via']==16) | (users['registered_via']==13), 'registered_via'] = 9

  # creates age slices an push odd values to unknown age
  users.loc[(users['bd']<=0) | (users['bd']>80),'bd'] = 0
  users.loc[(users['bd']<=15) & (users['bd']>0),'bd'] = 1
  users.loc[(users['bd']<=20) & (users['bd']>15),'bd'] = 2
  users.loc[(users['bd']<=25) & (users['bd']>20),'bd'] = 3
  users.loc[(users['bd']<=30) & (users['bd']>25),'bd'] = 4
  users.loc[(users['bd']<=35) & (users['bd']>30),'bd'] = 5
  users.loc[(users['bd']<=40) & (users['bd']>35),'bd'] = 6
  users.loc[(users['bd']<=45) & (users['bd']>40),'bd'] = 7
  users.loc[(users['bd']<=50) & (users['bd']>45),'bd'] = 8
  users.loc[(users['bd']<=60) & (users['bd']>50),'bd'] = 9
  users.loc[users['bd']>60,'bd'] = 10

  # guess age from registration channel
  users.loc[(users['bd']==0) & (users['registered_via']==3),'bd'] = 2
  users.loc[(users['bd']==0) & (users['registered_via']==4),'bd'] = 1
  users.loc[(users['bd']==0) & (users['registered_via']==7),'bd'] = 5
  users.loc[(users['bd']==0) & (users['registered_via']==9),'bd'] = 4
  
  # add a 'timediff' column to users set, made of the time difference between expiration and registration
  users = users.assign(timediff=pd.Series(np.zeros(len(users),dtype=float)).values)
  init = pd.to_datetime(users['registration_init_time'], format='%Y%m%d')
  last = pd.to_datetime(users['expiration_date'], format='%Y%m%d')
  users['timediff'] = last - init
  users['timediff'] = users['timediff'].dt.days
  
  # Label encoding and  32-bits unsigned int conversion, except for 'timediff'
  for feat in features.keys():
    if feat == 'timediff':
      # normalization to the max value
      users['timediff'] = users['timediff'] / users['timediff'].max()
    else:
      users[feat] = le.fit_transform(users[feat]) 
      users[feat] = users[feat].astype(np.uint32)
  
  # Write preprocessed file to disk
  filename = utils.namer(key='users', suffix='_prep.csv')
  users.to_csv(path_or_buf=filename, index=False)
  print("users_prep file written")

  return None


############################################
def pp_train(trainname='data/train.csv', testname='data/test.csv', features={}):
  """ feat eng. and cleaning of the train/test datafile and rewrite the result 
  as a new csv file.
  
  Parameters
  ==========
  trainname: string
    location and name of the training file.
    
  testname: string
    location and name of the testing file.

  features: python dict
    A dictionnary holding the feature name and the needed array sizes for binary encoding.
    The latter is not used here, we only need the feature names from the dict.

  key: str
    String holding the python dict key of the dataset to be worked on.
    
  Returns
  =======
  None
  """
  
  data = {'train': 0,
          'test': 0,}
  data['train'] = utils.dataload(filename=trainname)
  data['test'] = utils.dataload(filename=testname)
  
  trainsize = len(data['train'])
  le = LabelEncoder()

  # Apply preprocessing to test nad train sets
  for feat in features.keys():
    # Concatenate the 2 data files for a shared treatment
    tempdata = data['train'][feat].append(data['test'][feat])
    tempdata.fillna('Unknown', inplace=True) # Fill NaN by 'Unknown' cat.
    
    # Label encoding and  32-bits unsigned int conversion
    tempdata = le.fit_transform(tempdata)
    data['train'][feat] = tempdata[0:trainsize].astype(np.uint32)
    data['test'][feat] = tempdata[trainsize:].astype(np.uint32)
    del tempdata
    
  # Creates the matching files (for users/songs data) and kick useless fields
  for item in data.keys():
    if settings.match_build[item] == True:
      print('Building the %s matching files ...' % item)
      ids_matcher(train=data[item], key=item)
    else:
      print('Read the %s unmatched file' % item)

    # Extract the labels to a separate file
    if item == 'train':
      labels = data['train']['target'].astype(np.uint8).as_matrix()
      filename = utils.namer(key='labels', suffix='_prep.csv')
      labels.tofile(filename, sep=',', format='%u')
      data['train'].drop('target', axis=1, inplace=True)

    # Drop fields that are not used anymore to reduce memory usage (strings fields)
    data[item].drop('msno', axis=1, inplace=True)
    data[item].drop('song_id', axis=1, inplace=True)
    
    # Write preprocessed files to disk
    filename = utils.namer(key=item, suffix='_prep.csv')
    data[item].to_csv(path_or_buf=filename, index=False)
    print("%s_prep file written" % item)
  
  # Drop useless fields from users and songs data now
  filename = utils.namer(key='users', suffix='_prep.csv')
  users = utils.dataload(filename=filename)
  users.drop('msno', axis=1, inplace=True)
  
  filename = utils.namer(key='songs', suffix='_prep.csv')
  songs = utils.dataload(filename=filename)
  songs.drop('song_id', axis=1, inplace=True)
    
  # Write preprocessed files to disk
  filename = utils.namer(key='users', suffix='_prep.csv')
  users.to_csv(path_or_buf=filename, index=False)
  filename = utils.namer(key='songs', suffix='_prep.csv')
  songs.to_csv(path_or_buf=filename, index=False)
    
  print("-> Final pre-processed and cleaned files written")
  
  return None


############################################
def ids_matcher(train=None, key='train'):
  """ Creates files for songsid/msno variable matching between
  the songs/users and the train/test datafiles. Must be run
  for test and train sets separately.
  
  Parameters
  ==========
  train: pandas dataframe
    The dataframe holding either Train or test data.
    
  key: str
    String holding the python dict key of the dataset to be worked on.
  
  Returns
  =======   
  None
  """
  
  if settings.TIMER==True: time_start = time.time()

  file_users = utils.namer(key='users', suffix='_prep.csv')
  file_songs = utils.namer(key='songs', suffix='_prep.csv')
  
  users = pd.read_csv(file_users, usecols=['msno'])
  songs = pd.read_csv(file_songs, usecols=['song_id'])
  
  trainsize = len(train)
  le = LabelEncoder()

  # Do a common Label-Encoding for easy indexing, then split back datasets
  tempmsno = train['msno'].append(users['msno'])
  tempmsno = le.fit_transform(tempmsno)
  train['msno'] = tempmsno[0:trainsize]
  users['msno'] = tempmsno[trainsize:]
  del tempmsno
  
  tempmsongid = train['song_id'].append(songs['song_id'])
  tempmsongid = le.fit_transform(tempmsongid)
  train['song_id'] = tempmsongid[0:trainsize]
  songs['song_id'] = tempmsongid[trainsize:]
  del tempmsongid
  
  # Sort users/songs dataframes by increasing msno/songid label  
  users = users.sort_values('msno')
  songs = songs.sort_values('song_id')
  usersmax = max(users['msno'])
  songsmax = max(songs['song_id'])

  match = {'songs': [],
           'users': []}
  unmatch = {'songs': [],
             'users': []}
  # Fills the match array by the row in users/songs file that matches the i-th train entry
  print('It will take a while, go grab a coffee.')
  for i in range(trainsize):
    le_users = train.loc[i]['msno']
    name_users = users.loc[users['msno']==le_users].index
    if len(name_users) > 0:
      match['users'].append(name_users[0])
    else:
      match['users'].append(-1)
      unmatch['users'].append(i)
      if settings.MONITOR==True: print('No users match for %s data %d' % (key, i) )
    
    le_songs = train.loc[i]['song_id']
    name_songs = songs.loc[songs['song_id']==le_songs].index
    if len(name_songs) > 0:
      match['songs'].append(name_songs[0])
    else:
      match['songs'].append(-1)
      unmatch['songs'].append(i)
      if settings.MONITOR==True: print('No songs match for %s data %d' % (key, i) )
    
  print('%d training examples have no users matching' % len(unmatch['users']) )
  print('%d training examples have no songs matching' % len(unmatch['songs']) )

  # Write the matching files to disk
  filename = utils.namer(key=key, suffix='_matchedusers.csv')
  data = np.asarray(match['users'], dtype='int32')
  data.tofile(filename, sep=',', format='%u')
  
  filename = utils.namer(key=key, suffix='_matchedsongs.csv')
  data = np.asarray(match['songs'], dtype='int32')
  data.tofile(filename, sep=',', format='%u')
  
  filename = utils.namer(key=key, suffix='_unmatchedusers.csv')
  data = np.asarray(unmatch['users'], dtype='int32')
  data.tofile(filename, sep=',', format='%u')
  
  filename = utils.namer(key=key, suffix='_unmatchedsongs.csv')
  data = np.asarray(unmatch['songs'], dtype='int32')
  data.tofile(filename, sep=',', format='%u')
  
  if settings.TIMER==True: print("-> Train file builder = %0.2f s \n" % (time.time() - time_start))
  
  return None


############################################
def pp_all():
  """ feat eng. and cleaning of datafiles depending on the given switch.
  
  Parameters
  ==========
  None
    
  Returns
  =======
  None
  """
  
  if settings.TIMER==True: time_start = time.time()
  
  # Users data
  if settings.prep_switch['users'] == True:
    pp_users(filename=settings.path_users, features=settings.maxcat['users'])
  else:
    print('USERS pre-processed file will be read from disk.') 

  # Songs data
  if settings.prep_switch['songs'] == True:
    pp_songs(filename=settings.path_songs, features=settings.maxcat['songs'])
  else:
    print('SONGS pre-processed file will be read from disk.') 

  # Train and test data
  if settings.prep_switch['train'] == True:
    pp_train(trainname=settings.path_train, testname=settings.path_test, features=settings.maxcat['train'])
  else:
    print('TRAIN and TEST pre-processed files will be read from disk.') 
      
  if settings.TIMER==True: print("-> Pre-processing = %0.2f s \n" % (time.time() - time_start))  
  
  return None 


############################################
def binary(key='users'):
  """ Binary encoding of data matrix. Also calculate maxlabels and num_feat values
  for global dicts 'maxcat_*' and 'num_feat' through maxfiller() call.
  
  Parameters
  ==========
  key: str
    String holding the python dict key of the dataset to be worked on.
    
  Returns
  =======
  data: numpy ndarray
    The binarized data as a 2D matrix (row=entries, columns=feat.)
  """

  if settings.TIMER==True: time_start = time.time()
  
  # Converts dataframes to dict of matrices
  filename = utils.namer(key=key, suffix='_prep.csv')
  featdict = utils.data2mat(filename=filename, key=key)
  
  # Calculate number of cat. needed for Binanry Encoding, and total num. of feat. yielded
  utils.maxfiller(key=key, featdict=featdict)
  
  # Build the binarized file
  if settings.binarize_build == True:
    print('Building the %s binary file ...' % key)
    
    # Converts features to binary-encoded features
    tempdict={}
    for item in featdict.keys():
      if item == 'timediff':
        tempdict[item] = np.zeros(shape=[settings.num_data[key], 1], dtype='float32')
        for i in range(settings.num_data[key]):
          tempdict[item][i] = featdict[item][i]
      else:
        tempdict[item] = np.zeros(shape=[settings.num_data[key], settings.maxcat[key][item]], dtype='uint8')
        for i in range(settings.num_data[key]):
          tempdict[item][i,:] = utils.int2bin(featdict[item][i], settings.maxcat[key][item])
    
    # Put back all the features together into a single data matrix
    tempdata = []
    for item in featdict.keys():
      tempdata.append(tempdict[item])
      
    data = np.hstack(tempdata)
    
    # Save binary matrix as csv file
    filename = utils.namer(key=key, suffix='_binary.csv')
    data.tofile(filename, sep=',', format='%f')
    
  else:
    # Load the binarized file if it exists
    filename = utils.namer(key=key, suffix='_binary.csv')
    binfile = Path(filename)
    if binfile.exists():
      data = np.fromfile(filename, dtype='float32', count=-1, sep=',')
      data = np.reshape(data,[settings.num_data[key], settings.num_feat[key]])
      print('%s binarized file will be read from disk (%d x %d matrix).' % (key,
                                                                            settings.num_data[key],
                                                                            settings.num_feat[key]) )
    else:
      print('No binary file exist nor has been built for %s !' % key)
    
  if settings.TIMER==True: print("-> Binary Encoding = %0.2f s \n" % (time.time() - time_start))

  return data


