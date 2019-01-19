import os
from pathlib import Path

p = Path(__file__).parents[1]

OUTPUT_DIR_TRAIN=os.path.abspath(os.path.join(p, '..', 'data/raw/ml-1m/train.dat'))
OUTPUT_DIR_TEST=os.path.abspath(os.path.join(p, '..', 'data/raw/ml-1m/test.dat'))
ROOT_DIR=os.path.abspath(os.path.join(p, '..', 'data/raw/ml-1m/ratings.dat'))

NUM_USERS=6040
NUM_TEST_RATINGS=10


def count_rating_per_user():
    
    rating_per_user={}

    for line in open(ROOT_DIR):
        
        line=line.split('::')
        user_nr=int(line[0])
        
        if user_nr in rating_per_user:
            rating_per_user[user_nr]+=1                       
        else:
            rating_per_user[user_nr]=1

    return rating_per_user
            

  
def train_test_split():
    
    user_rating=count_rating_per_user()
    test_counter=0
    
    next_user=1
    
    train_writer=open(OUTPUT_DIR_TRAIN, 'w')
    test_writer=open(OUTPUT_DIR_TEST, 'w')
    
    for line in open(ROOT_DIR):
        
        splitted_line=line.split('::')
        user_nr=int(splitted_line[0])
        
        if user_rating[user_nr]<=NUM_TEST_RATINGS*2:
            next_user+=1
            continue
        
        try:
            if user_nr==next_user:
                write_test_samples=True
                next_user+=1
                
            if write_test_samples==True:
                test_writer.write(line)
                test_counter+=1
                
                if test_counter>=NUM_TEST_RATINGS:
                    test_counter=0
                    write_test_samples=False        
            else:
                train_writer.write(line)
        
        except KeyError:   
            print('Key not found')
            continue
              
if __name__ == "__main__":
    
    train_test_split()