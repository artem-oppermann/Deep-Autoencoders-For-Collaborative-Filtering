# Deep-Autoencoders-For-Collaborative-Filtering

Collaborative Filtering is a method used by recommender systems to make predictions about an interest of an specific user by collecting taste or preferences information from many other users. The technique of Collaborative Filtering has the underlying assumption that if a user A has the same taste or opinion on an issue as the person B, A is more likely to have B’s opinion on a different issue. 

In this project I predict the ratings a user would give a movie based on this user's taste and the taste of other users who watched and rated the same and similar movies.



## Datasets

The current version support only the MovieLens ml-1m.zip dataset obtained from https://grouplens.org/datasets/movielens/. 

## How to Use

- Download the ml-1m.zip dataset from  https://grouplens.org/datasets/movielens/.
- Devide the ```ratings.dat``` file from ml-1m.zip into training and testing datasets ```train.dat``` and ```test.dat```. Save them under ROOT_DIR/ml-1m/..
- Use shell to make TF_Record files out of the both ```train.dat``` and ```test.dat``` files by executing the command: 

        python tf_record_writer.py ROOT_DIR OUTPUT_DIR_TRAIN OUTPUT_DIR_TEST
      
- Use shell to start the training by executing the command (optionally parse your hyperparameters):

        python train.py \
               --tf_records_train_path=OUTPUT_DIR_TRAIN \
               --tf_records_test_path=OUTPUT_DIR_TEST \

## Training

During the training after each epoch the loss on the training and testing data set is shown. The loss is a root mean sqaurred error loss (RMSE).The RMSE represents the sample standard deviation of the differences between predicted values and observed values. E.g. a RMSE of 0.923 means that on an average the predicted rating deviates from the actual rating by 0.923 stars.
Here are the first 50 epochs of the training.

    epoch_nr: 0, train_loss: 1.169, test_loss: 1.020
    epoch_nr: 1, train_loss: 1.000, test_loss: 1.018
    epoch_nr: 2, train_loss: 1.003, test_loss: 1.027
    epoch_nr: 3, train_loss: 1.000, test_loss: 1.004
    epoch_nr: 4, train_loss: 0.989, test_loss: 0.995
    epoch_nr: 5, train_loss: 0.982, test_loss: 0.991
    epoch_nr: 6, train_loss: 0.974, test_loss: 0.988
    epoch_nr: 7, train_loss: 0.963, test_loss: 0.975
    epoch_nr: 8, train_loss: 0.955, test_loss: 0.969
    epoch_nr: 9, train_loss: 0.947, test_loss: 0.962
    epoch_nr: 10, train_loss: 0.936, test_loss: 0.959
    epoch_nr: 11, train_loss: 0.930, test_loss: 0.951
    epoch_nr: 12, train_loss: 0.923, test_loss: 0.954
    epoch_nr: 13, train_loss: 0.916, test_loss: 0.950
    epoch_nr: 14, train_loss: 0.912, test_loss: 0.941
    epoch_nr: 15, train_loss: 0.906, test_loss: 0.939
    epoch_nr: 16, train_loss: 0.902, test_loss: 0.935
    epoch_nr: 17, train_loss: 0.898, test_loss: 0.935
    epoch_nr: 18, train_loss: 0.893, test_loss: 0.928
    epoch_nr: 19, train_loss: 0.891, test_loss: 0.928
    epoch_nr: 20, train_loss: 0.889, test_loss: 0.931
    epoch_nr: 21, train_loss: 0.886, test_loss: 0.933
    epoch_nr: 22, train_loss: 0.884, test_loss: 0.929
    epoch_nr: 23, train_loss: 0.882, test_loss: 0.932
    epoch_nr: 24, train_loss: 0.879, test_loss: 0.928
    epoch_nr: 25, train_loss: 0.879, test_loss: 0.936
    epoch_nr: 26, train_loss: 0.879, test_loss: 0.926
    epoch_nr: 27, train_loss: 0.877, test_loss: 0.928
    epoch_nr: 28, train_loss: 0.875, test_loss: 0.923
    epoch_nr: 29, train_loss: 0.874, test_loss: 0.923
    epoch_nr: 30, train_loss: 0.873, test_loss: 0.923
    epoch_nr: 31, train_loss: 0.871, test_loss: 0.922
    epoch_nr: 32, train_loss: 0.869, test_loss: 0.927
    epoch_nr: 33, train_loss: 0.867, test_loss: 0.922
    epoch_nr: 34, train_loss: 0.871, test_loss: 0.921
    epoch_nr: 35, train_loss: 0.867, test_loss: 0.927
    epoch_nr: 36, train_loss: 0.866, test_loss: 0.925
    epoch_nr: 37, train_loss: 0.862, test_loss: 0.923
    epoch_nr: 38, train_loss: 0.860, test_loss: 0.926
    epoch_nr: 39, train_loss: 0.859, test_loss: 0.924
    epoch_nr: 40, train_loss: 0.859, test_loss: 0.925
    epoch_nr: 41, train_loss: 0.857, test_loss: 0.923
    epoch_nr: 42, train_loss: 0.855, test_loss: 0.928
    epoch_nr: 43, train_loss: 0.853, test_loss: 0.924
    epoch_nr: 44, train_loss: 0.852, test_loss: 0.928
    epoch_nr: 45, train_loss: 0.851, test_loss: 0.928
    epoch_nr: 46, train_loss: 0.850, test_loss: 0.928
    epoch_nr: 47, train_loss: 0.847, test_loss: 0.928
    epoch_nr: 48, train_loss: 0.846, test_loss: 0.925
    epoch_nr: 49, train_loss: 0.844, test_loss: 0.928
    epoch_nr: 50, train_loss: 0.844, test_loss: 0.929
