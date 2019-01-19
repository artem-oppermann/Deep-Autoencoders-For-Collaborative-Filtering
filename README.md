# Deep-Autoencoders-For-Collaborative-Filtering

Collaborative Filtering is a method used by recommender systems to make predictions about an interest of an specific user by collecting taste or preferences information from many other users. The technique of Collaborative Filtering has the underlying assumption that if a user A has the same taste or opinion on an issue as the person B, A is more likely to have B’s opinion on a different issue. 

In this project I predict the ratings a user would give a movie based on this user's taste and the taste of other users who watched and rated the same and similar movies.



## Datasets

The current version support only the MovieLens ml-1m.zip dataset obtained from https://grouplens.org/datasets/movielens/. 

## How to Use

- Download the ml-1m.zip dataset from  https://grouplens.org/datasets/movielens/.
- Devide the ```ratings.dat``` file from ml-1m.zip into training and testing datasets ```train.dat``` and ```test.dat```. by using the command 

       python src\data\train_test_split.py 
          
- Use shell to make TF_Record files out of the both ```train.dat``` and ```test.dat``` files by executing the command: 

       python src\data\tf_record_writer.py 
      
- Use shell to start the training by executing the command (optionally parse your hyperparameters):

        python training.py 

## Training

During the training after each epoch the loss on the training and testing data set is shown. The loss is a root mean sqaurred error loss (RMSE).The RMSE represents the sample standard deviation of the differences between predicted values and observed values. E.g. a RMSE of 0.923 means that on an average the predicted rating deviates from the actual rating by 0.923 stars.
Here are the first 50 epochs of the training.


    
    

