import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, Span, Token
nlp = spacy.load("en_core_web_md")
import re
from train_test import pipeline
from itertools import chain

#TODO create config file

def read_files(essays_path, adus_path):
    # INPUTS 
    essays = pd.read_csv(essays_path)
    adus = pd.read_csv(adus_path)  
    return essays, adus 


def essays_process():
    essays, adus = read_files("../data/output_csv/essays.csv" , "../data/output_csv/adus.csv")
    
    #TODO remove the lines below, for now is just for testing  
    n = 50
    split_pct = 0.8
    essays = essays.sample(n)
    train = essays.sample(frac=split_pct)
    test = essays.drop(train.index)
    
    print(len(train))
    print(len(test))
    pipeline(train,test, adus)
    
essays_process()