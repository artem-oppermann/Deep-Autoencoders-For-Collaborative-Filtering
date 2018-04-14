import numpy as np
import tensorflow as tf
import sys
from preprocess_data import _get_dataset

#OUTPUT_DIR_TRAIN='C:/Users/Admin/Desktop/deep_learning_data/colaborative_filtering/TFRecords_normal_ratings/tf_records_1M/train'
#OUTPUT_DIR_TEST='C:/Users/Admin/Desktop/deep_learning_data/colaborative_filtering/TFRecords_normal_ratings/tf_records_1M/test'


def _add_to_tfrecord(data_sample,tfrecord_writer):
    
    data_sample=list(data_sample.astype(dtype=np.float32))
    
    example = tf.train.Example(features=tf.train.Features(feature={'movie_ratings': float_feature(data_sample)}))                                          
    tfrecord_writer.write(example.SerializeToString())
    

def _get_output_filename(output_dir, idx, name):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def main():
    ''' Writes the .txt training and testing data into binary TF_Records.'''

    SAMPLES_PER_FILES=100
    
    training_set, test_set=_get_dataset(sys.argv[1])

    for data_set, name, dir_ in zip([training_set, test_set], ['train', 'test'], [sys.argv[2], sys.argv[3]]):
        
        num_samples=len(data_set)
        i = 0
        fidx = 1

        while i < num_samples:
           
            tf_filename = _get_output_filename(dir_, fidx,  name=name)
            
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                
                j = 0
                
                while i < num_samples and j < SAMPLES_PER_FILES:
                    
                    sys.stdout.write('\r>> Converting sample %d/%d' % (i+1, num_samples))
                    sys.stdout.flush()
    
                    sample = data_set[i]
                    _add_to_tfrecord(sample, tfrecord_writer)
                    
                    i += 1
                    j += 1
                fidx += 1

    print('\nFinished converting the dataset!')

    
    
    
if __name__ == "__main__":

    #main(output_dir=[OUTPUT_DIR_TRAIN,OUTPUT_DIR_TEST])
    main()
            
    







