##Sklearn Models
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import warnings
import pickle
import datetime
from preprocessing import create_training_data
warnings.filterwarnings('ignore')

st = datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')

f = open(f'../data/output/reports/Research-Case-Results-{st}.txt', 'w')
f.write('Research Case Result Report \n')
#Settings

#classifiers = ['logistic_regression', 'random_forest', 'naive_bayes', 'xgboost', 'svm' ]
classifiers = ['logistic_regression']

models = {'logistic_regression': LogisticRegression(solver='newton-cg'),
        'random_forest': RandomForestClassifier(),  
        'xgboost': XGBClassifier(), 
        'svm': SVC()  }

#segmentations = ['sentence', 'paragraph', 'n_grams', 'clause', 'constituency1', 'gold_standard']3
#segmentations = ['sentence', 'constituency1', 'gold_standard']
segmentations = ['gold_standard']


                
classifications = ['multiclass']
#classifications = ['binary', 'multiclass', 'two_binary']

def pipeline(train, test):

    f.write(f'\n We are using {len(train)} essays \n')


    f.write(f'\n Classifiers are using {classifiers} \n')
    f.write(f'\n Segmentations are using {segmentations} \n')
    f.write(f'\n Classifications are using {classifications} \n')

    f.write(f'\n Start Pipeline \n')
    
    for segmentation_mode in segmentations:
        print(segmentation_mode)
        
        f.write(f'\nSegmentation Type: {segmentation_mode} \n')
        
        text_segmentation(segmentation_mode,train, test)
        
    f.close()

def text_segmentation(segmentation_mode,train, test):
            
    X_train, y_train_adu, y_train_clpr, error_vector_dict_train, error_mean_dict_train = create_training_data(train, segmentation_mode= segmentation_mode)
    X_test, y_test_adu, y_test_clpr, error_vector_dict_test, error_mean_dict_test = create_training_data(test, segmentation_mode= segmentation_mode)
    
    f.write(f"Segmentation Errors Vectors {segmentation_mode}\n")  

    f.write("\nTrain_Error_Vectors\n")
    for k,v in error_vector_dict_train.items():
        f.write(f"{k}: {v}\n")
    f.write("\nTest_Error_Vectors\n")
    for k,v in error_vector_dict_test.items():
        f.write(f"{k}: {v}\n")

    f.write(f"\nSegmentation Error Means {segmentation_mode}\n")  
    f.write("\nTrain\n")
    for k,v in error_mean_dict_train.items():
        f.write(f"{k}: {v}\n")
    f.write("\nTest\n")
    for k,v in error_mean_dict_test.items():
        f.write(f"{k}: {v}\n")

    for classification in classifications:
        print(classification)
        
        f.write(f'\n Classification Type: {classification} \n')

        classification_type(classification, 
                            X_train, y_train_adu,y_train_clpr, 
                            X_test, y_test_adu, y_test_clpr)


def classification_type(classification, X_train, y_train_adu, y_train_clpr, X_test, y_test_adu, y_test_clpr):
    
    if classification == 'binary':
        
        for classifier in classifiers:
            train_test_classifer(classification, classifier, X_train, y_train_adu, X_test, y_test_adu)
    
    if classification  == 'multiclass':

        for classifier in classifiers:
            train_test_classifer(classification, classifier, X_train, y_train_clpr, X_test, y_test_clpr)


    if classification == 'two_binary':

        for classifier in classifiers:
            two_binary_classification(classification, classifier,  X_train, y_train_adu, y_train_clpr, X_test, y_test_adu, y_test_clpr)
            

def two_binary_classification(classification, first_classifier, X_train, y_train_adu, y_train_clpr, X_test, y_test_adu, y_test_clpr):
    
    second_classifiers = ['logistic_regression']
    
    print(f'First Classier: {first_classifier}')
    f.write(f'\n First Classier: {first_classifier} \n')
    
    cl1 = train_test_classifer(classification, first_classifier, 
                               X_train, y_train_adu, 
                               X_test, y_test_adu, True)
    for cli_1 in cl1:
        preds_cl1 = cli_1.predict(X_test)
        preds_cl1_adu_index = np.where(preds_cl1=='ADU')

        X_test_cl1_pred_adu = X_test[preds_cl1_adu_index]
        y_test_cl1_pred_adu = y_test_adu[preds_cl1_adu_index]


        for second_classifier in second_classifiers:
            print(f'Second Classier: {second_classifier}')
            
            f.write(f'\n Second Classier: {second_classifier} \n')
        
            clpr_index_train = np.where(y_train_clpr!='Non-ADU')[0]
            clpr_index_test = np.where(y_test_clpr!='Non-ADU')[0]

            X_train_clpr_only = X_train[clpr_index_train].copy()
            X_test_clpr_only = X_test[clpr_index_test].copy()


            y_train_clpr_only = y_train_clpr[clpr_index_train].copy()

            y_test_clpr_only = y_test_clpr[clpr_index_test].copy()

            cl2 = train_test_classifer(classification, second_classifier, X_train_clpr_only,  y_train_clpr_only, X_test, y_test_clpr, True)

            for cli_2 in cl2:

                preds_cl2 = cli_2.predict(X_test_cl1_pred_adu)

                preds_all = preds_cl1.copy()
                preds_all[preds_cl1_adu_index] = preds_cl2
                preds_all 

                print(classification_report(preds_all, y_test_clpr))

                f.write(f'\n Classification Report: {classification_report(preds_all, y_test_clpr)} \n')

                
def train_test_classifer(classification, classifier, X_train, y_train, X_test, y_test, multiclass = False):
    
        print(classifier)
        f.write(f'\n Model {classifier}: \n')
        list_models = []
        
        if classifier in models:
            simple_model = simple_models(models[classifier],X_train, y_train)
        
        grid_model = globals()[classifier](X_train, y_train)
        
        list_models.append(simple_model)
        list_models.append(grid_model)
        
        for i, modeli in enumerate(list_models): 
            
            ###Write the Report in the file 
            preds_model = modeli.predict(X_test)
            if multiclass:
                write_results(classification_report(y_test, preds_model), True, modeli)
            else:
                write_results(classification_report(y_test, preds_model))
            
            ###Saving the models as pickle 
            time = datetime.datetime.now().strftime('%H%M%S')
            with open(f'../data/output/models/{classification}_{classifier}_{i}_{time}.bin', 'wb') as f_out:
                pickle.dump(modeli, f_out) # write final_model in .bin file
                f_out.close()
        
        return list_models

def simple_models(model_selected, X_train, y_train):
    
    model = model_selected
    model.fit(X_train, y_train)
    
    return model
    
def logistic_regression(X_train, y_train):

    ####################################################
    # parameter grid
    parameters = {
        'penalty' : ['l1','l2'], 
        'C'       : np.logspace(-3,3,7),
        'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
    }
    
    logreg_grid = GridSearchCV(LogisticRegression(), 
                       param_grid=parameters,
                       scoring='accuracy',
                       cv = 10)
    
    logreg_grid.fit(X_train,y_train)        
       
    return logreg_grid

            
def random_forest(X_train, y_train):
           
    ####################################################
    # parameter grid
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 2, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    
    return rf_random
            
def naive_bayes(X_train, y_train):
    
    #NB doesn't have any hyperparameters to tune.
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    gnb_models = []
    gnb_models.append(gnb)
    
    return gnb_models
            
def xgboost(X_train, y_train):
               
    ####################################################
    # parameter grid
    param_grid = {
    "max_depth": [3, 4, 5, 7],
    "learning_rate": [0.1, 0.01, 0.05],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.5],
    }

    # Init classifier
    xgb_cl = XGBClassifier()
    # Init Grid Search
    grid_xgb = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3)
    xgb_grid = grid_xgb.fit(X_train, y_train)
        
    return xgb_grid


def svm(X_train, y_train):
            
    ####################################################
    # parameter grid
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    svm_grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

    # fitting the model for grid search
    svm_grid.fit(X_train, y_train)
    
    return svm_grid
    
    
def write_results(classification_report, grid = False, model = None): 
      
    if grid:
        print("Tuned Hyperparameters: ", model.best_params_)
        print("Accuracy: ", model.best_score_)      
        f.write(f'\n Tuned Hyperparameters: {model.best_params_} \n')
        f.write(f'\n Accuracy: {model.best_score_} \n')
        
    print(classification_report)    
    f.write(f'\n Classification Report: \n {classification_report} \n')