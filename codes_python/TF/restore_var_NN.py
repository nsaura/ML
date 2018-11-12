# To fetch some variables in a pre trained NN

# From https://towardsdatascience.com/deploy-tensorflow-models-9813b5a705d5
# Code git 
# https://github.com/FrancescoSaverioZuppichini/TensorFlow-Serving-Example/blob/master/serve.py

#### Notes incomplètes voir le site directement

import tensorflow as tf
import os

SAVE_PATH = './save'
MODEL_NAME = 'test'
VERSION = 1
SERVE_PATH = './serve/{}/{}'.format(MODEL_NAME, VERSION)

checkpoint = tf.train.latest_checkpoint(SAVE_PATH)

tf.reset_default_graph()

with tf.Session() as sess:
    # import the saved graph
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    # get the graph for this session
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    # get the tensors that we need
    inputs = graph.get_tensor_by_name('inputs:0')
    predictions = graph.get_tensor_by_name('prediction/Sigmoid:0')
    
    
# We need to build the tensor info from them that will be used to create the signature definition 
# that will be passed to the SavedModelBuilder instance

# create tensors info
model_input = tf.saved_model.utils.build_tensor_info(inputs)
model_output = tf.saved_model.utils.build_tensor_info(predictions)

#Now we can finally create the model signature that identifies what the serving is going to expect from the client

# build signature definition
signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'inputs': model_input},
    outputs={'outputs': model_output},
    method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    

