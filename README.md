# Deep-Autoencoders-For-Collaborative-Filtering

Collaborative Filtering is a method used by recommender systems to make predictions about an interest of an specific user by collecting taste or preferences information from many other users. The technique of Collaborative Filtering has the underlying assumption that if a user A has the same taste or opinion on an issue as the person B, A is more likely to have B’s opinion on a different issue. 

In this project I predict the ratings a user would give a movie based on this user's taste and the taste of other users who watched and rated the same and similar movies.



## Datasets

The current version support only the MovieLens dataset obtained from https://grouplens.org/datasets/movielens/. 

How to Use

- Download the ml-1m.zip dataset from  https://grouplens.org/datasets/movielens/ into a directory.
- Devide the ratings.dat file into a train.dat and test.dat and save them under ROOT_DIR/ml-1m.zip/
- In preprocess_data.py change the ROOT_DIR path
- Use tf_record_writer.py to transfort the train.dat and test.dat to TF_Records files (change the OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST paths   which are the output directories for the TF_Records files)
- execute train.py to start the training (change tf_records_train_path, tf_records_test_path)
