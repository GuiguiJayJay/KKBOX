import numpy as np
import tensorflow as tf
import time

import kklib.utils as utils
import kklib.settings as settings


############################################
def encoder(x, weights):
  """ Encoder for Auto-Encoder.
  
  Parameters
  ==========
  x: TensorFlow placeholder
    A placeholder holding the features (which are also our target 
    for an Auto-Encoder).
    
  weights: TensorFlow Variable
    A variable holding the weights for this layer.
    
  Returns
  =======
  layer_1: TensorFlow neural network
    A graph node storing the operation to be performed when called.
  """

  layer_1 = tf.nn.sigmoid(tf.matmul(x, weights))
  
  return layer_1


############################################
def decoder(x, weights):
  """ Decoder for Auto-Encoder. Note the matrix used here is the transpose
  of the matrix used for the Encoder.
  
  Parameters
  ==========
  x: TensorFlow placeholder
    A placeholder holding the features (which are also our target 
    for an Auto-Encoder).
    
  weights: TensorFlow Variable
    A variable holding the weights for this layer.
    
  Returns
  =======
  layer_1: TensorFlow neural network
    A graph node storing the operation to be performed when called.
  """

  layer_1 = tf.nn.sigmoid(tf.matmul(x, tf.transpose(weights)))
  
  return layer_1


############################################
def AE_model(init_lr=0.1,
             decay_rate=0.95,
             batchsize=1,
             epochs=1,
             display_steps=10000,
             num_hidden=20,
             data={},
             key='users'):          
  """ Runs the specified Auto-Encoder.
  
  Parameters
  ==========
  init_lr: float
    The starting learning rate.
    
  decay_rate: float
    The decay rate.
    
  batchsize: int
    The size of each data batch.
    
  epochs: int
    The number of epochs (full data cycle).
    
  display_steps: int
    Every "display_step", a short summary of the progress is sent to the terminal.
    
  num_hidden: int
    Number of hidden units of the auto-encoder.
      
  data: python dict
    Dictionary holding the feature names and associated data.
    
  key: str
    String holding the python dict key of the dataset to be worked on.
    
  Returns
  =======
  None
  """
                
  # Some variables/parameters declaration
  learning_rate = tf.placeholder(tf.float32, shape=[])
  steps = int(settings.num_data[key]/batchsize) + 1
  
  with tf.name_scope('Parameters'):
    X = tf.placeholder('float', [None, settings.num_feat[key]])
    weights = tf.Variable(tf.random_normal([settings.num_feat[key], num_hidden],mean=0,stddev=0.5))
  
  # Predictions and labels definition
  with tf.name_scope('Model'):
    predictions = decoder(encoder(X,weights),weights)
    preds = tf.round(predictions) # round to integers the predictions (-> accuracy calc.)
  with tf.name_scope('Target'):
    labels = X

  # Define accuracy and objective
  with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.reduce_max(tf.abs(labels - predictions), keep_dims=True, axis=1))
  with tf.name_scope('Accuracy'):
    results = tf.reduce_sum(tf.cast(tf.equal(labels,preds), tf.int32), keep_dims=True, axis=1)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(results, settings.num_feat[key]), tf.int32), axis=0) / tf.size(results)

  # Define optimization
  with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
  # Create a Saver to save NN weights
  saver = tf.train.Saver([weights])

  # Add variables to be written to a summary
  tf.summary.scalar('loss', loss)
  summary = tf.summary.merge_all()
  
  # Create a Python dict to hold various TensorFlow objects, and Training parameters
  tfdict = {'loss': loss,
            'accuracy': accuracy,
            'optimizer': optimizer,
            'summary': summary,
            'saver': saver,
            'X': X,
            'learning_rate': learning_rate,
            'weights': weights}
            
  params = {'init_lr': init_lr,
            'decay_rate': decay_rate,
            'batchsize': batchsize,
            'epochs': epochs,
            'display_steps': display_steps,
            'steps': steps}
  
  # train the auto-encoder to generate weights
  if settings.ML_mode['AE'] == True:
    print('Start training Auto-Encoder for %s set' % key)
    AE_training(params = params,
                data = data,
                tfobjects = tfdict,
                key = key)
  # Load weights from disk
  else:
    print('Use weights for %s A-E from the disk' % key)
    AE_load(tfobjects = tfdict,
            key = key)
  
  return None


############################################
def AE_training(params={},
                data={},
                tfobjects={},
                key='users'): 
  """ The training loop for the Auto-Encoder.
  
  Parameters
  ==========
  params: python dict
    Dictionnary holding training parameters and their names
      'init_lr' (float): The starting learning rate.
      'decay_rate' (float): The decay rate.
      'batchsize' (int): The size of each data batch.
      'epochs' (int): The number of epochs (full data cycle).
      'display_steps' (int): Every "display_step", a short summary 
                             of the progress is sent to the terminal.
      'steps'(int): The number of steps per epoch.
    
  data: python dict
    Dictionary holding the feature names and associated data.
    
  tfobjects: python dict
    Dictionary holding TensorFlow objects and their names.

  key: str
    String holding the python dict key of the dataset to be worked on.
    
  Returns
  =======
  None
  """
  
  start_time_tot = time.time()
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('train/', sess.graph)
      
    # Misc. var.
    start_time_part = start_time_tot
    lossvec = np.zeros(shape=[params['display_steps']], dtype='float')
    prevloss = 1
    
    for j in range(0,params['epochs']):
      for i in range(0,params['steps']):
        # Get a batch from dataset
        batch_x = utils.batch_AE(step = i,
                                 batchsize = params['batchsize'],
                                 data = data,
                                 key = key)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l, summary = sess.run([tfobjects['optimizer'],
                                  tfobjects['loss'],
                                  tfobjects['summary']],
                                 feed_dict={tfobjects['X']: batch_x,
                                            tfobjects['learning_rate']: params['init_lr']})
        lossvec[i%params['display_steps']] = l
        summary_writer.add_summary(summary, i*(j+1))

        # Display result every "display_steps" steps
        if i == 0:
          print('Loss at epoch %i - step %i: %f' % (j,i,l))
          print('Learning rate: %0.4f' % params['init_lr'])
            
        elif i % params['display_steps'] == 0:
          l = lossvec.sum()/params['display_steps']
          print('Loss at epoch %i - step %i: %f (runtime = %0.2f sec.)' % (j,i,l,time.time() - start_time_part))
          start_time_part = time.time()
          
          # Adjust learning if loss progress by less than 5%
          if (l/prevloss) > 0.95:
              params['init_lr'] = params['init_lr']*params['decay_rate']
              print('Learning rate: %0.4f' % params['init_lr'])
          prevloss = l
            
        elif i == params['steps']-1:
          l = lossvec.sum()/params['display_steps']
          print('Loss at epoch %i - step %i: %f (runtime = %0.2f sec.)' % (j,i,l,time.time() - start_time_part))
          
    # Compute total accuracy after training
    batch_x = utils.batch_AE(step = 0,
                             batchsize = settings.num_data[key],
                             data = data,
                             key = key)      
    acc = tfobjects['accuracy'].eval({tfobjects['X']: batch_x})
    print()
    print('Total accuracy: %0.3f %%' % (100*acc))
    print("-> %s Auto-Encoder = %0.2f s \n" % (key, time.time()-start_time_tot) )
    
    # Save the NN weights
    filename =  'weights/'+ key + '/AE_weights'
    tfobjects['saver'].save(sess, filename, global_step=i)

  # Reset the graph or the loop over the datasets will not work
  tf.reset_default_graph()
  # End session to avoid any (more) memory problems
  sess.close()

  return None
  
  
############################################
def AE_load(tfobjects={},
            key='users'): 
  """ Use saved Auto-Encoder coefficients to make predictions.
  
  Parameters
  ==========  
  tfobjects: python dict
    Dictionary holding TensorFlow objects and their names.
    
  Returns
  =======
  None
  """
  
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
      
    # Load the NN weights
    filename =  'weights/'+ key + '/AE_weights-' + str(settings.num_data[key])
    tfobjects['saver'].restore(sess, filename)
    
    # Write the raw NN weights to a global variable
    settings.NNweightsdict[key] = tfobjects['weights'].eval()
    settings.num_feat[key] = np.shape(settings.NNweightsdict[key])[1]
    
    print('\t %s encoded features = %d' % (key, settings.num_feat[key]) )

  # Reset the graph or the loop over the datasets will not work
  tf.reset_default_graph()
  # End session to avoid any (more) memory problems
  sess.close()

  return None
  
  
############################################
def NeuralNetwork(x, num_hid1=0, num_hid2=0, num_hid3=0, initializer=None):  
  """ Definition of a generic 3-layers Neural Network.
  
  Parameters
  ==========
  x: TensorFlow placeholder
    A placeholder holding the features.
    
  num_hid'i': int
    The number of hidden units for the i-th hidden layer.
    
  initializer: tf initializer
    A particluar initializer to set starting weights of each layers.
    
  Returns
  =======
  output: 
    The output layer of the NN (pre-activation).
  """
  
  # hidden layer 1
  layer_1 = tf.layers.dense(x,
                            num_hid1, 
                            kernel_initializer = initializer,
                            bias_initializer = initializer)
  # hidden layer 2
  layer_2 = tf.layers.dense(layer_1,
                            num_hid2,
                            kernel_initializer = initializer,
                            bias_initializer = initializer,
                            activation = tf.nn.tanh)
  # hidden layer 3
  layer_3 = tf.layers.dense(layer_2,
                            num_hid2,
                            kernel_initializer = initializer,
                            bias_initializer = initializer)
  # hidden layer 4
  layer_4 = tf.layers.dense(layer_3,
                            num_hid3,
                            kernel_initializer = initializer,
                            bias_initializer = initializer,
                            activation = tf.nn.tanh)
  # Output layer (pre-activation)
  out_layer = tf.layers.dense(layer_4,
                              1,
                              kernel_initializer = initializer,
                              bias_initializer = initializer)
  
  return out_layer

  
############################################
def NN_model(init_lr=0.1,
             decay_rate=0.95,
             batchsize=1,
             epochs=1,
             display_steps=10000,
             num_hidden1=40,
             num_hidden2=20,
             num_hidden3=20,
             data={},
             matching={},
             unmatching={}):          
  """ Runs the specified Neural-Network.
  
  Parameters
  ==========
  init_lr: float
    The starting learning rate.
    
  decay_rate: float
    The decay rate.
    
  batchsize: int
    The size of each data batch.
    
  epochs: int
    The number of epochs (full data cycle).
    
  display_steps: int
    Every "display_step", a short summary of the progress is sent to the terminal.
    
  num_hidden1(2): int
    Number of hidden units of the 1st(2nd) layer.
      
  data: python dict
    Dictionary holding the feature names and associated data.
    
  matching: python dict
    Dictionnary holding datanames ('train' or 'test') and the associated python dict
    which holds 2 matching arrays and their names ('songs' and 'users').
    
  unmatching: python dict
    Dictionnary holding datanames ('train' or 'test') and the associated data entries
    for which no data where found in songs/users.
    
  Returns
  =======
  None
  """
                
  # Some variables/parameters declaration
  learning_rate = tf.placeholder(tf.float32, shape=[])
  steps = int(settings.num_data[settings.trainingmode]/batchsize)
  
  num_feat = settings.num_feat['users'] + settings.num_feat['songs'] + settings.num_feat[settings.trainingmode]
  
  X = tf.placeholder('float32', [None, num_feat])
  labels = tf.placeholder('float32', [None,1])
  
  initializer = tf.random_normal_initializer(mean=0.0, stddev=0.5, dtype=tf.float32)
  logits = NeuralNetwork(X, num_hidden1, num_hidden2, num_hidden3, initializer)
  
  # Predictions and labels definition
  with tf.name_scope('Model'):
    pred_probas = tf.sigmoid(logits)
    pred_classes = tf.greater(pred_probas,0.5) # returns bool

  # Define accuracy and objective
  with tf.name_scope('Loss'):
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
    loss = tf.reduce_mean(tf.square(pred_probas-labels))
    correct = tf.equal(pred_classes, tf.equal(labels,1.0))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float')) 
    
  # Define optimization
  with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  # Create a Saver to save NN weights
  saver = tf.train.Saver()
  
  # Add variables to be written to a summary
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('accuracy', accuracy)
  summary = tf.summary.merge_all()
  
  # Create a Python dict to hold various TensorFlow objects, and Training parameters
  tfdict = {'loss': loss,
            'accuracy': accuracy,
            'optimizer': optimizer,
            'summary': summary,
            'saver': saver,
            'X': X,
            'labels': labels,
            'pred_probas': pred_probas,
            'pred_classes': pred_classes,
            'learning_rate': learning_rate}
            
  params = {'init_lr': init_lr,
            'decay_rate': decay_rate,
            'batchsize': batchsize,
            'epochs': epochs,
            'display_steps': display_steps,
            'steps': steps}
  
  # Train the NN to generate weights
  if settings.ML_mode['NN'] == True:
    print('Start training Neural Network')
    NN_training(params = params,
                data = data,
                matching = matching,
                unmatching = unmatching,
                tfobjects = tfdict)
  # Load weights from disk
  else:
    print('Use weights for the Neural Network from the disk')
    NN_load(tfobjects = tfdict,
            data = data,
            matching = matching,
            unmatching = unmatching,
            params = params)
  
  return None
  
  
############################################
def NN_training(params={},
                data={},
                matching={},
                unmatching={},
                tfobjects={}): 
  """ The training loop for the Neural Network.
  
  Parameters
  ==========
  params: python dict
    Dictionnary holding training parameters and their names
      'init_lr' (float): The starting learning rate.
      'decay_rate' (float): The decay rate.
      'batchsize' (int): The size of each data batch.
      'epochs' (int): The number of epochs (full data cycle).
      'display_steps' (int): Every "display_step", a short summary 
                             of the progress is sent to the terminal.
      'steps'(int): The number of steps per epoch.
    
  data: python dict
    Dictionary holding the feature names and associated data.
    
  matching: python dict
    Dictionnary holding datanames ('train' or 'test') and the associated python dict
    which holds 2 matching arrays and their names ('songs' and 'users').
    
  unmatching: python dict
    Dictionnary holding datanames ('train' or 'test') and the associated data entries
    for which no data where found in songs/users.
    
  tfobjects: python dict
    Dictionary holding TensorFlow objects and their names.
    
  Returns
  =======
  None
  """
  
  start_time_tot = time.time()
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(init)
    summary_writer = tf.summary.FileWriter('train/', sess.graph)
      
    # Misc. var.
    start_time_part = start_time_tot
    lossvec = np.zeros(shape=[params['display_steps']], dtype='float')
    prevloss = 1
    acc = 0
        
    for j in range(0,params['epochs']):
      prevacc = acc
      acc = 0
      order = np.random.permutation(settings.num_data['train']) # random permutation of data
      for i in range(0,params['steps']):
        # Build a batch from datasets
        batch_x, target = utils.batch_full(step = i,
                                           batchsize = params['batchsize'],
                                           data = data,
                                           matching = matching,
                                           unmatching = unmatching,
                                           order = order)
        
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l, summary = sess.run([tfobjects['optimizer'],
                                  tfobjects['loss'],
                                  tfobjects['summary']],
                                 feed_dict={tfobjects['X']: batch_x,
                                            tfobjects['labels']: target,
                                            tfobjects['learning_rate']: params['init_lr']})
        lossvec[i%params['display_steps']] = l
        summary_writer.add_summary(summary, i*(j+1))

        # Display result every "display_steps" steps
        if i == 0:
          print('Learning rate: %0.4e' % params['init_lr'])
          
        elif i % params['display_steps'] == 0:
          l = lossvec.sum()/params['display_steps']
          print('Loss at epoch %i - step %i: %f (runtime = %0.2f sec.)' % (j,i,l,time.time() - start_time_part))
          start_time_part = time.time()
          # Adjust learning if loss progress by less than few %
          if (l/prevloss) > 0.99:
              params['init_lr'] = params['init_lr']*params['decay_rate']
              print('Learning rate: %0.4f' % params['init_lr'])
          prevloss = l
          
        elif i == params['steps']-1:
          l = lossvec.sum()/params['display_steps']
          print('Loss at epoch %i - step %i: %f (runtime = %0.2f sec.)' % (j,i,l,time.time() - start_time_part))

          
      # Compute total accuracy after each epoch in several steps (computer crashes else)
      print('Calculating accuracy ....')
      for k in range(0,params['steps']):
        # Build a batch from datasets
        batch_x, target = utils.batch_full(step = k,
                                           batchsize = params['batchsize'],
                                           data = data,
                                           matching = matching,
                                           unmatching = unmatching,
                                           order = order)   
        tempacc = tfobjects['accuracy'].eval({tfobjects['X']: batch_x, tfobjects['labels']: target})
        acc += tempacc
      acc = acc/k
      print('Total accuracy: %0.2f %%' % (100*acc))
      print("-> Neural Network running time = %0.2f s \n" % (time.time()-start_time_tot) )

      # Adjust learning rate depending on how accuracy progresses
      if prevacc > acc - 0.001:
        params['init_lr'] = params['init_lr']*params['decay_rate']
        
    # Save the NN weights
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    filename =  'weights/NN_weights'
    tfobjects['saver'].save(sess, filename, global_step=i)
    
  # Reset the graph or the loop over the datasets will not work
  tf.reset_default_graph()
  # End session to avoid any (more) memory problems
  sess.close()

  return None
  
  
############################################
def NN_load(tfobjects={},
            data={},
            matching={},
            unmatching={},
            params={},): 
  """ Use saved Neural Network coefficients to make predictions.
  
  Parameters
  ==========  
  tfobjects: python dict
    Dictionary holding TensorFlow objects and their names.
    
  data: python dict
    Dictionary holding the feature names and associated data.
    
  matching: python dict
    Dictionnary holding datanames ('train' or 'test') and the associated python dict
    which holds 2 matching arrays and their names ('songs' and 'users').
    
  unmatching: python dict
    Dictionnary holding datanames ('train' or 'test') and the associated data entries
    for which no data where found in songs/users.
    
  params: python dict
    Dictionnary holding training parameters and their names
      'init_lr' (float): The starting learning rate.
      'decay_rate' (float): The decay rate.
      'batchsize' (int): The size of each data batch.
      'epochs' (int): The number of epochs (full data cycle).
      'display_steps' (int): Every "display_step", a short summary 
                             of the progress is sent to the terminal.
      'steps'(int): The number of steps per epoch.
    
  Returns
  =======
  None
  """
  
  start_time_tot = time.time()
  
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(init)
    
    # Load the NN weights
    filename =  'weights/NN_weights-737740'
    tfobjects['saver'].restore(sess, filename)
    
    # Compute total accuracy on training set
    if settings.trainingmode == 'train':
      # Set the order as the index to file correctly the prediction file
      order = np.arange(0, settings.num_data['train'], 1, dtype=int)
      
      print('Calculating accuracy ....')
      for k in range(0,params['steps']):
        # Build a batch from datasets
        batch_x, target = utils.batch_full(step = k,
                                           batchsize = params['batchsize'],
                                           data = data,
                                           matching = matching,
                                           unmatching = unmatching,
                                           order = order)   
        tempacc = tfobjects['accuracy'].eval({tfobjects['X']: batch_x, tfobjects['labels']: target})
        acc += tempacc
      acc = acc/k  
      print('Total accuracy: %0.2f %%' % (100*acc))
      print("-> Neural Network running time = %0.2f s \n" % (time.time()-start_time_tot) )
      
    # Make predictions on test set
    else:
      # Set the order as the index to file correctly the prediction file
      order = np.arange(0, settings.num_data['test'], 1, dtype=int)
      
      print('Calculating predictions ....')
      predictions = []
      for k in range(0,params['steps']):
        start_time_part = time.time()
        
        # Build a batch from datasets
        batch_x, _ = utils.batch_full(step = k,
                                      batchsize = params['batchsize'],
                                      data = data,
                                      matching = matching,
                                      unmatching = unmatching,
                                      order = order)   
        pred_classes = tfobjects['pred_classes'].eval({tfobjects['X']: batch_x})
        for i in range(len(pred_classes)):
          if pred_classes[i] == True:
            predictions.append(1.0)
          else:
            predictions.append(0.0)
          
        if k % params['display_steps'] == 0:
          print('step %i (runtime = %0.2f sec.)' % (k,time.time() - start_time_part))
        
      # edit the sample_submission file to write preds
      predfile = utils.dataload(filename='data/sample_submission.csv')
      predfile['target'] = predictions
      predfile.to_csv(path_or_buf='data/predictions.csv', index=False)
      
    print("-> Neural Network running time = %0.2f s \n" % (time.time()-start_time_tot) )

  # Reset the graph or the loop over the datasets will not work
  tf.reset_default_graph()
  # End session to avoid any (more) memory problems
  sess.close()

  return None
  
  
  
