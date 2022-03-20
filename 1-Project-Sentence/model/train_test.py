from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import warnings
import pickle
import datetime
from preprocessing import create_units_from_docs, calculate_segmentation_accuracy
import config
warnings.filterwarnings('ignore')

st = datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')

f = open(f'../data/output/reports/Research-Case-Results-{st}.txt', 'w')
f.write('Research Case Result Report \n')

classifiers = config.classifiers
segmentations = config.segmentations
classifications = config.classifications
error_functions = config.error_function

models = {'logistic_regression': LogisticRegression(solver='newton-cg'),
        'random_forest': RandomForestClassifier(),  
        'xgboost': XGBClassifier(), 
        'svm': SVC(),
        'naive_bayes': GaussianNB()  }

def pipeline(train, test):

    f.write(f'\n We are using {len(train)} essays \n')


    f.write(f'\n Classifiers are using {classifiers} \n')
    f.write(f'\n Segmentations are using {segmentations} \n')
    f.write(f'\n Classifications are using {classifications} \n')

    f.write(f'\n Start Pipeline \n')
    
    for segmentation_mode in segmentations:
        print(segmentation_mode)
        
        f.write(f'\nSegmentation Type: {segmentation_mode} \n')
        
        units_train = create_units_from_docs(train, segmentation_mode= segmentation_mode)
        units_test = create_units_from_docs(test, segmentation_mode= segmentation_mode)
          
        X_features = config.span_features
    
        X_train = np.array([unit2fv(unit, X_features) for unit in units_train])
        X_test = np.array([unit2fv(unit, X_features) for unit in units_test])

        for error_function in error_functions: 

            error_mean_dict_train = calculate_segmentation_accuracy(units_train, error_function)
            error_mean_dict_test = calculate_segmentation_accuracy(units_test, error_function)

            f.write(f"\nSegmentation Error Means\nSegmentation Mode: {segmentation_mode}\nError Function: {error_function} \n") 
            
            f.write("\nTrain\n")
            for k,v in error_mean_dict_train.items():
                f.write(f"{k}: {v}\n")
                
            f.write("\nTest\n")
            for k,v in error_mean_dict_test.items():
                f.write(f"{k}: {v}\n")        
        
            y_train_adu = create_y(units_train, label_mode= 'adu', error_function= error_function,  segmentation_mode = segmentation_mode )
            y_test_adu = create_y(units_test, label_mode= 'adu', error_function= error_function, segmentation_mode = segmentation_mode)

            y_train_clpr = create_y(units_train, label_mode= 'clpr', error_function= error_function,  segmentation_mode = segmentation_mode )
            y_test_clpr = create_y(units_test, label_mode= 'clpr', error_function= error_function, segmentation_mode = segmentation_mode)
            
            for classification in classifications:
        
                print(classification)
                
                f.write(f'\n Classification Type: {classification} \n')
                
                
                for classifier in classifiers:
                
                    f.write(f'\n Model: {classifier}, Error Function: {error_function}, Segmentation Mode: {segmentation_mode} \n')

                    
                    if classification == 'binary':  
                        
                        trained_models = train_models(classifier, X_train, y_train_adu)
                        
                        test_and_save_models(trained_models , classification, classifier, X_test, y_test_adu, error_function,  segmentation_mode)
                        
                    if classification  == 'multiclass':
                        
                        trained_models = train_models(classifier, X_train, y_train_clpr)
                        
                        test_and_save_models(trained_models , classification, classifier, X_test, y_test_clpr, error_function,  segmentation_mode)
                        
                    if classification  == 'two_binary':
                        two_binary_classification(classification, classifier, X_train, X_test, y_train_adu, y_test_adu, y_train_clpr, y_test_clpr, error_function , segmentation_mode)
                                    
    f.close()


            
def unit2fv(unit, feature_list):
    
    fv = np.array([unit._.get(feature) for feature in feature_list], dtype='object')
    
    _fv = np.array([np.reshape(feature, -1) for feature in fv], dtype='object')
    
    return np.concatenate(_fv)


def create_training_test_data(units, label_mode, error_function , segmentation_mode = None):

    threshold=0
    if segmentation_mode == 'n_grams': 
        threshold=0.5
   
    X_features = config.span_features
    
    X = np.array([unit2fv(unit, X_features) for unit in units])
    y = np.array([unit._.get_label(label_mode=label_mode, threshold= threshold, error_function = error_function) for unit in units])
    
    return X, y

def create_y(units, label_mode, error_function , segmentation_mode = None):
    
    threshold=0
    if segmentation_mode == 'n_grams': 
        threshold=0.5
        
    y = np.array([unit._.get_label(label_mode=label_mode, threshold= threshold, error_function = error_function) for unit in units])
    
    return y


def two_binary_classification(classification, classifier, X_train, X_test, y_train_adu, y_test_adu, y_train_clpr, y_test_clpr, error_function , segmentation_mode):
            
    
    cl1_trained_models = []
        
    cl1_trained_models = train_models(classifier, X_train, y_train_adu)
    
    for i, cl1_model in enumerate(cl1_trained_models): 
        
        ###Write the Report in the file 
        preds_cl1 = cl1_model.predict(X_test)
        
        preds_cl1_adu_index = np.where(preds_cl1=='ADU')
        X_test_cl1_pred_adu = X_test[preds_cl1_adu_index]

    
        f.write(f'\nClassifiers: {classifier}, Error Function: {error_function}, Segmentation Mode: {segmentation_mode} \n')

        clpr_index_train = np.where(y_train_clpr!='Non-ADU')[0]
                
        X_train_clpr_only = X_train[clpr_index_train].copy()
        
        y_train_clpr_only = y_train_clpr[clpr_index_train].copy()

        
        cl2_trained_models = []
        
        
        if i == 0:
            cl2_model = simple_models(models[classifier],X_train_clpr_only, y_train_clpr_only)
            
            
            
        elif i == 1: 
            cl2_model = globals()[classifier](X_train_clpr_only, y_train_clpr_only) 
        
        
        cl2_trained_models = train_models(classifier, X_train_clpr_only, y_train_clpr_only)

        preds_cl2 = cl2_model.predict(X_test_cl1_pred_adu)

        preds_all = preds_cl1.copy()
        preds_all[preds_cl1_adu_index] = preds_cl2
        preds_all 
        
        
        write_results(preds_all, y_test_clpr, i, cl2_model)
        
         ##Saving the models as pickle 
        time = datetime.datetime.now().strftime('%H%M%S')
        with open(f'../data/output/models/{classification}_Classifier_{classifier}_ErrorFunction_{error_function}_SegmentationMode_{segmentation_mode}_{i}_{time}.bin', 'wb') as f_out:
            pickle.dump(cl2_model, f_out) # write final_model in .bin file
            f_out.close()
            
    return cl2_trained_models  


def test_and_save_models(trained_models , classification, classifier, X_test, y_test, error_function,  segmentation_mode):
    
    for i, modeli in enumerate(trained_models): 
    
        ###Write the Report in the file 
        preds_model = modeli.predict(X_test)
        
        if i == 1:
            write_results(y_test, preds_model, True, modeli)
        else:
            write_results(y_test, preds_model)
        
        ###Saving the models as pickle 
        time = datetime.datetime.now().strftime('%H%M%S')
        with open(f'../data/output/models/{classification}_{classifier}_ErrorFunction_{error_function}_SegmentationMode_{segmentation_mode}_{i}_{time}.bin', 'wb') as f_out:
            pickle.dump(modeli, f_out) # write final_model in .bin file
            f_out.close()
            
                        
def train_models(classifier, X_train, y_train):
    
    list_models = []  
    if classifier in models:
        simple_model = simple_models(models[classifier],X_train, y_train)
        list_models.append(simple_model)

    # grid_model = globals()[classifier](X_train, y_train) 
    # list_models.append(grid_model)
        
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
    gnb_model = GaussianNB()
    gnb_model.fit(X_train, y_train)
        
    return gnb_model
            
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
    
    
def write_results(y_test, preds_model, grid = False, model = None): 
      
    if grid:     
        f.write(f'\n Tuned Hyperparameters: {model.best_params_} \n')
        f.write(f'\n Accuracy: {model.best_score_} \n')
      
    f.write(f'\n Confusion Matrix: \n {confusion_matrix(y_test, preds_model)} \n')    
    
    f.write(f'\n Classification Report: \n {classification_report(y_test, preds_model)} \n')
    
