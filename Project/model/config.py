#Preprocessing Settings 

doc_features = ['num_tokens', 'para_starts']
span_features = ['word_emb', 'sent_emb', 'num_tokens', 'num_verbs', 'num_pos_pronouns', 'num_conj_adv', 'num_punct', 'is_para_start',
                 'index_in_doc', 'num_claim_indicator', 'num_premise_indicator', 'has_question_mark', 'has_personal_pronoun',
                 'has_possessive_pronoun', 'has_modal_verb', 'is_first_token_gerund', 'tree_depth', 'contextual_features_prev' ,'contextual_features_next']

# getters that are not used as features
span_utilities = ['prev_unit', 'idx_start', 'idx_end', ]
# methods
span_methods = ['get_nth_unit', 'get_prev_unit_attr', 'get_label_and_error', 'get_label', 'get_possible_labels']
token_features =['word_emb']

extensions_dict = dict(doc_features=doc_features, span_features=span_features+span_utilities,
                    token_features=token_features, span_methods=span_methods)




#Train and test Settings 

#classifiers = ['logistic_regression', 'random_forest', 'naive_bayes', 'xgboost', 'svm' ]
classifiers = ['logistic_regression']

#segmentations = ['sentence', 'paragraph', 'n_grams', 'clause', 'constituency1', 'gold_standard']3
#segmentations = ['sentence', 'constituency1', 'gold_standard']
segmentations = ['sentence']

#classifications = ['binary', 'multiclass', 'two_binary'] 
classifications = ['binary']
