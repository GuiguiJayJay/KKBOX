
def init():
  """ All the running options switches are defined here,
  as well as the most commonly used variables, and some global variables.
  All of them have been set to global for possible further developpments.
  
  Parameters
  ==========
  None
  
  Returns
  =======
  None
  """
  ########################
  # OPTIONS
  ########################
  # Debugging tools
  global TIMER # displays time of every major step
  TIMER = True
  global MONITOR # displays monitoring infos
  MONITOR = True

  # Switches
  global prep_switch # Switch to preprocess each file
  prep_switch = {'users': False,
                 'songs': False,
                 'train': False} 
              
  # Files to build
  global match_build # Build a the users/songs matching files
  match_build = {'train': False,
                 'test': False}
  global binarize_build # Build the binarized files
  binarize_build = False
 
  # Modes
  global trainingmode # Mode of algorithm ('train' to use training set, 'test' for testing set)
  trainingmode = 'test'
  global bypassAE # Bypass the Auto-Encoder
  bypassAE = True
  global ML_mode # Software mode: True=train weights, False=load weights
  ML_mode = {'AE': False,
             'NN': False}


  ########################
  # GLOBAL VARIABLES
  ########################
  # Data paths
  global path_dir
  path_dir = 'data/'
  
  global path_users
  path_users =  path_dir + 'users.csv'
  
  global path_songs
  path_songs =  path_dir + 'songs.csv'
  
  global path_train
  path_train =  path_dir + 'train.csv'
  
  global path_test
  path_test = path_dir + 'test.csv'

  # Features to be processed, and binarization vectors sizes (if any)                
  global maxcat
  maxcat = {'users': {'city': 0,
                      'bd': 0,
                      'gender': 0,
                      'registered_via': 0,
                      'timediff': 0},
            'songs': {'song_length': 0,
                      'genre_ids': 0,
                      'artist_name': 0,
                      'language': 0},
            'train': {'source_system_tab': 0,
                      'source_screen_name': 0,
                      'source_type': 0},
            'test': {'source_system_tab': 0,
                     'source_screen_name': 0,
                     'source_type': 0} }
                     
  # Number of data entries for each datasets
  global num_data
  num_data = {'users': 0,
              'songs': 0,
              'train': 0,
              'test': 0 }
                     
  # Number of features for each datasets after OHE or LE
  global num_feat
  num_feat = {'users': 0,
              'songs': 0,
              'train': 0,
              'test': 0}
              
  # Tensor variables
  global NNweightsdict
  NNweightsdict = {'users': 0,
                   'songs': 0,
                   'train': 0,
                   'test': 0}
  
  
  
  
  
  
  
  
