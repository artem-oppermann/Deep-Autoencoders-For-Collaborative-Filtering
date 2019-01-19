import numpy as np
import tensorflow as tf
import os

from data.dataset import _get_training_data, _get_test_data
from model.train_model import TrainModel
from sklearn.metrics import mean_absolute_error, mean_squared_error


tf.app.flags.DEFINE_string('tf_records_train_path', 
                           os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/tf_records/train/')),
                           'Path of the training data.')

tf.app.flags.DEFINE_string('tf_records_test_path', 
                           os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/tf_records/test/')),
                           'Path of the test data.')

tf.app.flags.DEFINE_string('checkpoints_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'checkpoints/model.ckpt')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_integer('num_epoch', 1000,
                            'Number of training epochs.')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            'Size of the training batch.')

tf.app.flags.DEFINE_float('learning_rate',0.0005,
                          'Learning_Rate')

tf.app.flags.DEFINE_boolean('l2_reg', False,
                            'L2 regularization.'
                            )
tf.app.flags.DEFINE_float('lambda_',0.01,
                          'Wight decay factor.')

tf.app.flags.DEFINE_integer('num_v', 3952,
                            'Number of visible neurons (Number of movies the users rated.)')

tf.app.flags.DEFINE_integer('num_h', 128,
                            'Number of hidden neurons.)')

tf.app.flags.DEFINE_integer('num_samples', 5953,
                            'Number of training samples (Number of users, who gave a rating).')

FLAGS = tf.app.flags.FLAGS


def main(_):
    '''Building the graph, opening of a session and starting the training od the neural network.'''
    
    num_batches=int(FLAGS.num_samples/FLAGS.batch_size)

    with tf.Graph().as_default():

        train_data, train_data_infer=_get_training_data(FLAGS)
        test_data=_get_test_data(FLAGS)
        
        iter_train = train_data.make_initializable_iterator()
        iter_train_infer=train_data_infer.make_initializable_iterator()
        iter_test=test_data.make_initializable_iterator()
        
        x_train= iter_train.get_next()
        x_train_infer=iter_train_infer.get_next()
        x_test=iter_test.get_next()

        model=TrainModel(FLAGS, 'training')

        train_op, train_loss_op=model.train(x_train)
        prediction, labels, test_loss_op, mae_ops=model._validation_loss(x_train_infer, x_test)
        
        saver=tf.train.Saver()
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            train_loss=0
            test_loss=[]
            mae=[]

            for epoch in range(FLAGS.num_epoch):
                
                sess.run(iter_train.initializer)
                sess.run(iter_train_infer.initializer)
                sess.run(iter_test.initializer)

                for batch_nr in range(num_batches):
                    
                    _, loss_=sess.run((train_op, train_loss_op))
                    train_loss+=loss_
                
                for i in range(FLAGS.num_samples):
                    
                    pred, labels_, loss_, mae_=sess.run((prediction, labels, test_loss_op,mae_ops))

                    test_loss.append(loss_)
                    mae.append(mae_)
                    
                print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f, mean_abs_error: %.3f'
                      %(epoch,(train_loss/num_batches),np.mean(test_loss), np.mean(mae)))
                
                if np.mean(mae)<0.9:
                    saver.save(sess, FLAGS.checkpoints_path)

                train_loss=0
                test_loss=[]
                mae=[]
                    
if __name__ == "__main__":
    
    tf.app.run()
