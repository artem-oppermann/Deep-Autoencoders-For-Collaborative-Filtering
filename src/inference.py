# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:39:52 2019

@author: Artem Oppermann
"""

import tensorflow as tf
import os
from model.inference_model import InferenceModel


tf.app.flags.DEFINE_string('checkpoints_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'checkpoints/')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_string('export_path_base', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'model-export/')), 
                           'Directory where to export the model.')

tf.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model.')

tf.app.flags.DEFINE_integer('num_v', 3952,
                            'Number of visible neurons (Number of movies the users rated.)')

FLAGS = tf.app.flags.FLAGS

def run_inference():
    
    inference_graph=tf.Graph()
    
    with inference_graph.as_default():
        
        model=InferenceModel(FLAGS)

        input_data=tf.placeholder(tf.float32, shape=[None, 3952])  
        ratings=model.inference(input_data)
        
        saver = tf.train.Saver()

    
    with tf.Session(graph=inference_graph) as sess:
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints_path)   
        saver.restore(sess, ckpt.model_checkpoint_path)      

        # Save the model
        export_path = os.path.join(tf.compat.as_bytes(FLAGS.export_path_base),
                                   tf.compat.as_bytes('model_v_%s'%str(FLAGS.model_version)))
        
        
        
        print('Exporting trained model to %s'%export_path)
        
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        
        
        # create tensors info
        predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(input_data)
        predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(ratings)
            
        # build prediction signature
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputs': predict_tensor_inputs_info},
                outputs={'ratings': predict_tensor_scores_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )
            
            
        # save the model
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_ratings': prediction_signature
            })

        builder.save()
        


if __name__ == "__main__":
    
    run_inference() 







