import pandas as pd
from train_test import pipeline


#TODO create config file

def read_files(essays_path, adus_path):
    # INPUTS 
    essays = pd.read_csv(essays_path)
    adus = pd.read_csv(adus_path)  
    return essays, adus 


def essays_process():
    essays, adus = read_files("../data/input/essays.csv" , "../data/input/adus.csv")
    
    #TODO remove the lines below, for now is just for testing  
    n = 5
    split_pct = 0.8
    essays = essays.sample(n)
    train = essays.sample(frac=split_pct)
    test = essays.drop(train.index)
    
    pipeline(train,test)

print("Start Running.....")
essays_process()
print("Project Run Successfully")