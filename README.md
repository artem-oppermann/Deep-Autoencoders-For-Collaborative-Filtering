# Deep-Autoencoders-For-Collaborative-Filtering

Collaborative Filtering is a method used by recommender systems to make predictions about an interest of an specific user by collecting taste or preferences information from many other users. The technique of Collaborative Filtering has the underlying assumption that if a user A has the same taste or opinion on an issue as the person B, A is more likely to have B’s opinion on a different issue. 

In this project I predict the ratings a user would give a movie based on this user's taste and the taste of other users who watched and rated the same and similar movies.



## Datasets

The current version support only the MovieLens ml-1m.zip dataset obtained from https://grouplens.org/datasets/movielens/. 

## Model Training

- Download the ml-1m.zip dataset from  https://grouplens.org/datasets/movielens/.
- Devide the ```ratings.dat``` file from ml-1m.zip into training and testing datasets ```train.dat``` and ```test.dat```. by using the command 

       python src\data\train_test_split.py 
          
- Use shell to make TF_Record files out of the both ```train.dat``` and ```test.dat``` files by executing the command: 

       python src\data\tf_record_writer.py 
      
- Use shell to start the training by executing the command (optionally parse your hyperparameters):

        python training.py 

### Training Results 

During the training after each epoch the loss on the training and testing data set is shown. The loss is a root mean squared error loss (MSE). The mean absolute error (mean_abs_error) is a better metric to validate the performance however.mean_abs_error tells the differences between predicted ratings and true ratings. E.g. a mean_abs_error of 0.923 means that on an average the predicted rating deviates from the actual rating by 0.923 stars.

       epoch_nr: 0, train_loss: 1.421, test_loss: 0.967, mean_abs_error: 0.801
       epoch_nr: 1, train_loss: 0.992, test_loss: 0.961, mean_abs_error: 0.797
       epoch_nr: 2, train_loss: 0.987, test_loss: 0.962, mean_abs_error: 0.798
       epoch_nr: 3, train_loss: 0.981, test_loss: 0.965, mean_abs_error: 0.801
       epoch_nr: 4, train_loss: 0.969, test_loss: 0.974, mean_abs_error: 0.808
       epoch_nr: 5, train_loss: 0.949, test_loss: 0.988, mean_abs_error: 0.822
    
    
## Author 

- **Artem Oppermann**: Msc.Physics, Deep Learning Engineer, Co-Founder of [Deep Learning Academy](https://www.deeplearning-academy.com/)
