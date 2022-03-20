import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, Span, Token
import re
import benepar
from itertools import chain
from spacy.pipeline import Sentencizer
import config
import subprocess
import sys

try:
    nlp = spacy.load("en_core_web_lg")
except:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg" ])
    nlp = spacy.load("en_core_web_lg")
    
try:
    nlp_trf = spacy.load('en_core_web_trf', disable=['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
except:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_trf" ])
    nlp_trf = spacy.load('en_core_web_trf', disable=['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])

try:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

except:
    
    benepar.download('benepar_en3')
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    

para_splitter = Sentencizer(punct_chars=['\n'])
extensions_dict = config.extensions_dict
span_features = config.span_features
adus = pd.read_csv("../data/input/adus.csv")

def create_extensions(extensions_dict=None, force=True):
        
    # Features that take 'unit' as input refer to the segmentation, they do not work with just any span.
    # Property attributes
    # Store starting and ending indices of spans in the whole doc
    # 1 list per each document: [(s1_start, s1_end), (s2_start, s2_end),.., (sn_start, sn_end)]
    
    Doc.set_extension("units_index_list", default=[],force=True)
    
    # Store essay_id within doc
    Doc.set_extension("essay_id", default=None, force=True)

    
    # Feature Getters
    def get_possible_labels(unit, error_function='percentage_correctness'):
        """
        Inputs: unit

        Outputs: label for the unit and segmentation error

        """

        def overlap_case(unit_start, unit_end, adu_start, adu_end):
            if adu_start >= unit_start and adu_end <= unit_end:
                # Case 1, ADU is fully contained in UNIT
                return 1

            elif adu_start <= unit_start and adu_end <=unit_end and adu_end>=unit_start:

                # Case 2, ADU starts before UNIT, start(Left) of ADU is cut
                return 2

            elif adu_start >= unit_start and adu_end >= unit_end and adu_start<unit_end:

                # Case 3, ADU starts after UNIT, end(Right) of ADU is cut
                return 3

            elif adu_start < unit_start and adu_end > unit_end:

                # Case 4, ADU starts before UNIT and ends after UNIT, both sides of ADU are cut
                return 4

            else: 
                # ADU does not overlap with UNIT
                return False
            

        def percentage_correctness(unit, adu_start, adu_end, overlap_case):

            if overlap_case==2:
                adu_start = unit._.idx_start
            elif overlap_case==3:
                adu_end = unit._.idx_end
            elif overlap_case==4:
                adu_start = unit._.idx_start
                adu_end = unit._.idx_end

            adu = unit.doc.char_span(adu_start, adu_end, alignment_mode='expand')
            

            unit_ntokens = len(unit)
            adu_ntokens = len(adu)
            pct_correct = adu_ntokens/unit_ntokens
            return pct_correct

        def extended_accuracy(unit, adu_start, adu_end, overlap_case):
            # Compares number of tokens to get the the correct ADU in proportional with UNIT length

            if overlap_case==2:
                adu_start = unit._.idx_start
            if overlap_case==3:
                adu_end = unit._.idx_end
            adu = unit.doc.char_span(adu_start, adu_end, alignment_mode='expand')

            unit_ntokens = len(unit)
            adu_ntokens = len(adu)
            diff_ntokens = np.abs(unit_ntokens - adu_ntokens)

            return 1/((diff_ntokens+1)**(np.log2(diff_ntokens+1)/np.log2(unit_ntokens+1)))


        if error_function.lower() == 'percentage_correctness':
            err_func = percentage_correctness
        elif error_function.lower() == 'extended_accuracy':
            err_func = extended_accuracy
        
        unit_start = unit._.idx_start
        unit_end = unit._.idx_end

        essay_id = unit.doc._.essay_id

        # DataFrame containing ADUs indices & labels, filtered for current essay_id

        adus_doc = adus[adus['essay_id'] == essay_id]

        def segmentation_error(unit, adu_start, adu_end, overlap_case, error_function):
            
            adu = unit.doc.char_span(adu_start, adu_end, alignment_mode='expand')
            
            # positive value = too many tokens in segment, unit should be shorter (include less non-adu tokens)
            # negative value = too less tokens in segment, unit should be longer (include more adu tokens)
            
            left_tokens = adu.start - unit.start
            right_tokens = unit.end - adu.end
            
            if error_function.lower() == 'percentage_correctness':
                err_func = percentage_correctness
            elif error_function.lower() == 'extended_accuracy':
                err_func = extended_accuracy

            
            return (left_tokens, err_func(unit, adu_start, adu_end, overlap_case), right_tokens)

             
# returns: (ADU_Type, (left_error_tokens, err_func, right_error_tokens))
        label_and_error = [(row['ADU_type'], segmentation_error(unit, row['start_ind'],row['end_ind'], 
                          overlap_case(unit_start, unit_end,row['start_ind'], row['end_ind']), error_function),
                          #(row['start_ind'], row['end_ind'])
                           ) 
                         for row_ind, row in adus_doc.iterrows() 
                         if unit_start < row['end_ind'] and unit_end >= row['start_ind'] ]

            
        return label_and_error

    
    def get_label(unit, label_mode='clpr', threshold=0, error_function='percentage_correctness'):
        error_tuple = unit._.get_possible_labels(error_function=error_function)

        if len(error_tuple) == 0:
            return "Non-ADU"
        else:
            # Get position of label with maximum accuracy
            label_position = np.argmax([error[1] for label, error in error_tuple])
            label = ''
            if error_tuple[label_position][1][1] > threshold:
                if label_mode=='clpr':
                    label = error_tuple[label_position][0]
                elif label_mode=='adu':
                    label = 'ADU'
                    
            else:
                label = "Non-ADU"

            return label
        
    def get_label_and_error(unit, error_function='percentage_correctness', label_mode='clpr', threshold=0):
        error_tuple = unit._.get_possible_labels(error_function=error_function)

        if len(error_tuple) == 0:
            return ("Non-ADU", ())
        else:
            # Get position of label with maximum accuracy
            label_position = np.argmax([error[1] for label, error in error_tuple])
            if error_tuple[label_position][1][1] > threshold:
                if label_mode=='clpr':
                    assigned_label_and_error = (error_tuple[label_position][0], error_tuple[label_position][1])
                elif label_mode=='adu':
                    assigned_label_and_error = ('ADU', error_tuple[label_position][1])
                    
            else:
                assigned_label_and_error = ("Non-ADU", ())

            return assigned_label_and_error

    
    def get_idx_start(unit):
        return unit[0].idx
    
    def get_idx_end(unit):
        return unit[-1].idx  + len(unit[-1])
    
    
    def get_para_starts(doc):
        # Units starting with \n or preceding \n are considered as paragraph starts
        # if start is 0, start -1 goes back to the last token of the doc

        
        return [int(doc[start].text =='\n' or doc[start-1].text=='\n') for start, end in doc._.units_index_list]
    
    def get_is_para_start(unit):
        
        para_starts = unit.doc._.para_starts
        unit_ind = unit._.index_in_doc
        
        return para_starts[unit_ind]
    
    def get_has_personal_pronoun(unit):
        
        return 'PRP' in [token.tag_ for token in unit]
    
    def get_has_possessive_pronoun(unit):
        
        return 'PRP$' in [token.tag_ for token in unit]     
    
    def get_has_modal_verb(unit):
        
        return 'MD' in [token.tag_ for token in unit]            
    
    def get_word_emb(obj):
        return obj.vector

    def get_sent_emb(unit):
        
       trf_doc = nlp_trf(unit.text)
       return trf_doc._.trf_data.tensors[1][0]
        
    
    def get_num_tokens(obj):
        return len(obj)
    
    def get_num_verbs(span):
        return sum([1 for token in span if token.pos_ == "VERB"])

    def get_num_pos_pronouns(span):
        return sum([1 for token in span if token.tag_ == "PRP$"])

    def get_num_pron(span):
        return sum([1 for token in span if token.pos_ == "PRON"])
    
    def get_num_conj_adv(span):
        conj_advs = ['moreover', 'incidentally', 'next', 'yet', 'finally', 'then', 'for example', 'thus', 'accordingly', 'namely', 'meanwhile', 'that is', 'also', 'undoubtedly', 'all in all', 'lately', 'hence', 'still', 'therefore', 'in addition', 'indeed', 'again', 'so', 'nevertheless', 'besides', 'instead', 'for instance', 'certainly', 'however', 'anyway', 'further', 'furthermore', 'similarly', 'now', 'in conclusion', 'nonetheless', 'thereafter', 'likewise', 'otherwise', 'consequently']
        return sum([1 for adv in conj_advs if adv in span.text.lower()])
    
        
    def get_num_claim_indicator(span):
        claim_indicators = ["accordingly", "as a result", "consequently", "conclude that", "clearly", "demonstrates that", "entails", "follows that", "hence", "however", "implies", "in fact", "in my opinion", "in short", "in conclusion", "indicates that", "it follows that", "it is highly probable that", "it is my contention", "it should be clear that", "I believe", "I mean", "I think", "must be that", "on the contrary", "points to the conclusions", "proves that", "shows that", "so", "suggests that", "the most obvious explanation", "the point I’m trying to make", "therefore", "thus", "the truth of the matter", "to sum up", "we may deduce"]
        
        return sum([1 for c_indicator in claim_indicators if c_indicator in span.text.lower()])
    
    def get_num_premise_indicator(span):
        premise_indicators=["after all", "assuming that", "as", "as indicated by", "as shown", "besides", "because", "deduced", "derived from", "due to", "firstly", "follows from", "for", "for example", "for instance", "for one thing", "for the reason that", "furthermore", "given that", "in addition", "in light of", "in that", "in view of", "in view of the fact that", "indicated by", "is supported by", "may be inferred", "moreover", "owing to", "researchers found that", "secondly", "this can be seen from", "since", "since the evidence is", "what’s more", "whereas",]
        return sum([1 for p_indicator in premise_indicators if p_indicator in span.text.lower()])
    
    def get_is_first_token_gerund(span):
        
        return span[0].tag_ =='VBG'
    
    def get_has_question_mark(span):
        return '?' in span.text

    def get_num_punct(span):
        return sum([1 for token in span if token.tag_ == "."])
    
    def get_tree_depth(unit):
        depths = {}

        def walk_tree(node, depth):
            depths[node.orth_] = depth
            if node.n_lefts + node.n_rights > 0:
                return [walk_tree(child, depth + 1) for child in node.children]

        walk_tree(unit.root, 0)
        return max(depths.values())
    

    def get_index_in_doc(span):
        """Gets index of the segmented unit in the doc"""
        span_start = span.start

        # span end not used yet
        span_end = span.end

        # finds where span_start is in units_index_list [(s1_start, s1_end), (s2_start, s2_end),.., (sn_start, sn_end)]
        # returns the index of the corresponding span
        return np.where([span.start in range(start, end) for start, end in span.doc._.units_index_list])[0][-1]


    def get_prev_unit(span):

        return span._.get_nth_unit(span._.index_in_doc-1)
    
        
    def get_nth_unit(span, n):

        # Tuple containing the start and end index of the nth span
        span_index = span.doc._.units_index_list[n]

        # Return nth span
        return span.doc[span_index[0]: span_index[1]]

    def get_prev_unit_attr(span, attribute):

        return span._.prev_unit._.get(attribute)

    def get_contextual_features_prev(unit):
        contextual_features_names=['num_tokens','num_verbs','num_pos_pronouns','num_conj_adv','num_punct','is_para_start','num_claim_indicator','num_premise_indicator','has_question_mark','has_personal_pronoun','has_possessive_pronoun','has_modal_verb','is_first_token_gerund','tree_depth']
        
        contextual_features = np.array([])
        for feature in contextual_features_names:
            if unit._.index_in_doc==0:
                contextual_features = np.append(contextual_features,0)
            else:
                contextual_features = np.append(contextual_features, unit._.prev_unit._.get(feature))
        return contextual_features

    def get_contextual_features_next(unit):
        contextual_features_names=['num_tokens','num_verbs','num_pos_pronouns','num_conj_adv','num_punct','is_para_start','num_claim_indicator','num_premise_indicator','has_question_mark','has_personal_pronoun','has_possessive_pronoun','has_modal_verb','is_first_token_gerund','tree_depth']
        
        contextual_features = np.array([])

        try:
            next_unit = unit._.get_nth_unit(unit._.index_in_doc + 1)
        except:
            return [0 for feature in contextual_features_names]
        else:
            return [next_unit._.get(feature) for feature in contextual_features_names]

            
    
    
    # Iterate list of features and Set Extensions (Just to not manually set extensions one by one)
    
    for feature in extensions_dict['doc_features']:
        Doc.set_extension(feature, force=force, getter=locals()[f"get_{feature}"])
        
    for feature in extensions_dict['span_features']:
        Span.set_extension(feature, force=force, getter=locals()[f"get_{feature}"])
        
    for feature in extensions_dict['token_features']:
        Token.set_extension(feature, force=force, getter=locals()[f"get_{feature}"])
        
    for method in extensions_dict['span_methods']:
        Span.set_extension(method, force=force, method=locals()[method])


def segmentation(doc=None ,mode = 'sentence', n_grams=15):
    if mode=='paragraph':
        with nlp.select_pipes(disable=nlp.pipe_names):
            para_doc = para_splitter(nlp(doc.text))
            p_units = list(para_doc.sents)
            doc._.units_index_list = [(unit.start, unit.end) for unit in p_units]
            
            units = [doc[start:end] for start, end in doc._.units_index_list]
            
            return units
            
    elif mode=='sentence':
        # segment by sentences
        units = [sent for sent in doc.sents  if not (sent.text.isspace() or sent.text =='')] 
        
        # keep track of (start, end) of units in doc object
        doc._.units_index_list = [(unit.start, unit.end) for unit in units]
        return units
    
    elif mode =='n_grams':
        # Code to segment with 15 grams here (average)  
        units = [doc[i:i+n_grams] for i in range(len(doc))]

        doc._.units_index_list = [(unit.start, unit.end) for unit in units]

        return units
    
    elif mode=='clause':
        # Code to segment by clause
        pass
    elif mode=='constituency1':
        # Take the first level subordinating conjunction (SBAR)
        # The first dependent clause
        units = []
        for sent in doc.sents:
            for node in sent._.constituents:

                if "SBAR" in node._.labels:

                    # Before SBAR
                    units.append(sent.doc[sent.start:node.start])
                    # SBAR
                    units.append(sent.doc[node.start:node.end])

                    # After SBAR
                    units.append(sent.doc[node.end:sent.end])

                    # Break out to take only the first SBAR we encounter
                    break
        
        units = [unit for unit in units if unit.text != '']
        doc._.units_index_list = [(unit.start, unit.end) for unit in units]
        
        return units
        
    elif mode=='token':
        return [token for token in doc if not (token.text.isspace() or token.text =='')]
    elif mode=='gold_standard':
        
        # Segments ADUs according to annotations
        adu_inds = adus[adus['essay_id']==doc._.essay_id].sort_values('start_ind')[['start_ind','end_ind']]

        units = []

        start = 0
        for i, row in adu_inds.iterrows():

            # From previous adu end to current adu start (Non-ADU)
            end = row['start_ind']-1

            units.append(doc.char_span(start,end, alignment_mode='expand'))

            start = row['start_ind']
            end = row['end_ind']

            # From current adu start to current adu end
            units.append(doc.char_span(start,end,  alignment_mode='expand'))

            # set current adu end as start for next iteration
            start = row['end_ind']
        
        units = [unit for unit in units if unit.text != '']
        # keep track of (start, end) of units in doc object
        doc._.units_index_list = [(unit.start, unit.end) for unit in units]
        
        return units


def calculate_segmentation_accuracy(units, error_function='percentage_correctness'):
    
    
    
    start_errors = np.array([])
    segmentation_accs = np.array([])
    end_errors = np.array([])
    early_start_errors = np.array([])
    late_start_errors = np.array([])
    early_end_errors = np.array([])
    late_end_errors = np.array([])
    for unit in units:
        error_tuple = unit._.get_label_and_error(error_function=error_function)[1]
        
        if len(error_tuple) != 0:
            
            if error_tuple[0] < 0:
                late_start_errors = np.append(late_start_errors, error_tuple[0])
            elif error_tuple[0] > 0:
                early_start_errors = np.append(early_start_errors, error_tuple[0])
            
            segmentation_accs = np.append(segmentation_accs, error_tuple[1])
            
            if error_tuple[2] < 0:
                early_end_errors = np.append(early_end_errors, error_tuple[2])
            elif error_tuple[2] > 0:
                late_end_errors = np.append(late_end_errors, error_tuple[2])
            end_errors = np.append(end_errors, error_tuple[2])





#     start_error = sum((start_errors**2))/len(start_errors)

#     end_error = sum((end_errors**2))/len(end_errors)

#     segmentation_acc = segmentation_accs.mean()
    
    # error_vector_dict = dict(start_early_vector = early_start_errors, start_late_vector = late_start_errors, segmentation_accs_vector = segmentation_accs,
    #              end_early_vector = early_end_errors, end_late_vector = late_end_errors)
    
    error_mean_dict = dict(start_early = early_start_errors.mean(), start_late = late_start_errors.mean(),
                   segmentation_accs = segmentation_accs.mean(),end_early = early_end_errors.mean(),
                   end_late = late_end_errors.mean())
    
    return error_mean_dict



def create_units_from_docs(df, segmentation_mode='sentence', n_grams=15):
    
    # Run
    create_extensions(extensions_dict) 
    
    data = [(row['text'], dict(id=row['essay_id'])) for ind, row in df.iterrows()]
    docs = []
    
    if segmentation_mode != "constituency1":
    
        for doc, context in nlp.pipe(data, as_tuples=True, disable=['benepar']):
            doc._.essay_id = context['id']
            docs.append(doc)
    
    else:
        
        for doc, context in nlp.pipe(data, as_tuples=True):
            doc._.essay_id = context['id']
            docs.append(doc)

    segmented_docs = [segmentation(doc, mode=segmentation_mode ,n_grams=n_grams) for doc in docs]
    
    # Flatten lists (Dissolve docs boundaries and store all units together in one huge list)
    units = list(chain.from_iterable(segmented_docs))

    
    return units




