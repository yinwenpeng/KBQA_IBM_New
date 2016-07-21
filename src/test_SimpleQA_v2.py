import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
import random
import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN
import time

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from load_data import load_train, load_word2vec_to_init, load_word2id_char2id, load_test_or_valid
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Conv_with_input_para, Average_Pooling_for_SimpleQA, Average_Pooling_for_Top, create_conv_para, pythonList_into_theanoIntMatrix, Max_Pooling, cosine, pythonList_into_theanoFloatMatrix
from random import shuffle

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

from scipy import linalg, mat, dot

# from preprocess_wikiQA import compute_map_mrr

#need to change
'''


4) fine-tune word embeddings
5) translation bettween
6) max sentence length to 40:   good and best
7) implement attention by euclid, not cosine: good
8) stop words by Yi Yang
9) normalized first word matching feature
10) only use bleu1 and nist1
11) only use bleu4 and nist5



Doesnt work:
1) lr0.08, kern30, window=5, update10
8) kern as Yu's paper
7) shuffle training data: should influence little as batch size is 1
3) use bleu and nist scores
1) true sentence lengths
2) unnormalized sentence length
8) euclid uses 1/exp(x)
'''

def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, word_nkerns=500, char_nkerns=100, batch_size=1, window_width=3,
                    emb_size=500, char_emb_size=100, hidden_size=200,
                    margin=0.5, L2_weight=0.0003, update_freq=1, norm_threshold=5.0, max_truncate=40, 
                    max_char_len=40, max_des_len=20, max_relation_len=5, max_Q_len=30, train_neg_size=6, 
                    neg_all=100, train_size=75893, test_size=19835, mark='_BiasedMaxPool_lr0.1_word500_char10067.71'):  #train_size=75909, test_size=17386
#     maxSentLength=max_truncate+2*(window_width-1)
    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/freebase/SimpleQuestions_v2/'
    triple_files=['annotated_fb_data_train.entitylinking.top20_succSet_asInput.txt', 'annotated_fb_data_test.entitylinking.top20_succSet_asInput.fromMo.txt']

    rng = numpy.random.RandomState(23455)
    word2id, char2id=load_word2id_char2id(mark)
#     datasets, datasets_test, length_per_example_test, vocab_size, char_size=load_test_or_valid(triple_files[0], triple_files[1], max_char_len, max_des_len, max_relation_len, max_Q_len, train_size, test_size)#max_char_len, max_des_len, max_relation_len, max_Q_len

    datasets_test, length_per_example_test, word2id, char2id  = load_test_or_valid(triple_files[1], char2id, word2id, max_char_len, max_des_len, max_relation_len, max_Q_len, test_size)
    vocab_size=len(word2id)
    char_size=len(char2id)
    print 'vocab_size:', vocab_size, 'char_size:', char_size

#     train_data=datasets
#     valid_data=datasets[1]
    test_data=datasets_test
#     result=(pos_entity_char, pos_entity_des, relations, entity_char_lengths, entity_des_lengths, relation_lengths, mention_char_ids, remainQ_word_ids, mention_char_lens, remainQ_word_lens, entity_scores)
#     
#     train_pos_entity_char=train_data[0]
#     train_pos_entity_des=train_data[1]
#     train_relations=train_data[2]
#     train_entity_char_lengths=train_data[3]
#     train_entity_des_lengths=train_data[4]
#     train_relation_lengths=train_data[5]
#     train_mention_char_ids=train_data[6]
#     train_remainQ_word_ids=train_data[7]
#     train_mention_char_lens=train_data[8]
#     train_remainQ_word_len=train_data[9]
#     train_entity_scores=train_data[10]

    test_pos_entity_char=test_data[0]
    test_pos_entity_des=test_data[1]
    test_relations=test_data[2]
    test_entity_char_lengths=test_data[3]
    test_entity_des_lengths=test_data[4]
    test_relation_lengths=test_data[5]
    test_mention_char_ids=test_data[6]
    test_remainQ_word_ids=test_data[7]
    test_mention_char_lens=test_data[8]
    test_remainQ_word_len=test_data[9]
    test_entity_scores=test_data[10]
# 
#     test_pos_entity_char=test_data[0]       #matrix, each row for line example, all head and tail entity, iteratively: 40*2*51
#     test_pos_entity_des=test_data[1]        #matrix, each row for a examle: 20*2*51
#     test_relations=test_data[2]             #matrix, each row for a example: 5*51
#     test_entity_char_lengths=test_data[3]   #matrix, each row for a example: 3*2*51  (three valies for one entity)
#     test_entity_des_lengths=test_data[4]    #matrix, each row for a example: 3*2*51  (three values for one entity)
#     test_relation_lengths=test_data[5]      #matrix, each row for a example: 3*51
#     test_mention_char_ids=test_data[6]      #matrix, each row for a mention: 40
#     test_remainQ_word_ids=test_data[7]      #matrix, each row for a question: 30
#     test_mention_char_lens=test_data[8]     #matrix, each three values for a mention: 3
#     test_remainQ_word_len=test_data[9]      #matrix, each three values for a remain question: 3
    

#     train_sizes=[len(train_pos_entity_char), len(train_pos_entity_des), len(train_relations), len(train_entity_char_lengths), len(train_entity_des_lengths),\
#            len(train_relation_lengths), len(train_mention_char_ids), len(train_remainQ_word_ids), len(train_mention_char_lens), len(train_remainQ_word_len), len(train_entity_scores)]
#     if sum(train_sizes)/len(train_sizes)!=train_size:
#         print 'weird size:', train_sizes
#         exit(0)

    test_sizes=[len(test_pos_entity_char), len(test_pos_entity_des), len(test_relations), len(test_entity_char_lengths), len(test_entity_des_lengths),\
           len(test_relation_lengths), len(test_mention_char_ids), len(test_remainQ_word_ids), len(test_mention_char_lens), len(test_remainQ_word_len), len(test_entity_scores)]
    if sum(test_sizes)/len(test_sizes)!=test_size:
        print 'weird size:', test_sizes
        exit(0)

#     n_train_batches=train_size/batch_size
#     n_test_batches=test_size/batch_size
    
#     train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
#     test_batch_start=list(numpy.arange(n_test_batches)*batch_size)
    
#     indices_train_pos_entity_char=pythonList_into_theanoIntMatrix(train_pos_entity_char)
#     indices_train_pos_entity_des=pythonList_into_theanoIntMatrix(train_pos_entity_des)
#     indices_train_relations=pythonList_into_theanoIntMatrix(train_relations)
#     indices_train_entity_char_lengths=pythonList_into_theanoIntMatrix(train_entity_char_lengths)
#     indices_train_entity_des_lengths=pythonList_into_theanoIntMatrix(train_entity_des_lengths)
#     indices_train_relation_lengths=pythonList_into_theanoIntMatrix(train_relation_lengths)
#     indices_train_mention_char_ids=pythonList_into_theanoIntMatrix(train_mention_char_ids)
#     indices_train_remainQ_word_ids=pythonList_into_theanoIntMatrix(train_remainQ_word_ids)
#     indices_train_mention_char_lens=pythonList_into_theanoIntMatrix(train_mention_char_lens)
#     indices_train_remainQ_word_len=pythonList_into_theanoIntMatrix(train_remainQ_word_len)   
#     indices_train_entity_scores=pythonList_into_theanoFloatMatrix(train_entity_scores) 
    
#     indices_test_pos_entity_char=pythonList_into_theanoIntMatrix(test_pos_entity_char)
#     indices_test_pos_entity_des=pythonList_into_theanoIntMatrix(test_pos_entity_des)
#     indices_test_relations=pythonList_into_theanoIntMatrix(test_relations)
#     indices_test_entity_char_lengths=pythonList_into_theanoIntMatrix(test_entity_char_lengths)
#     indices_test_entity_des_lengths=pythonList_into_theanoIntMatrix(test_entity_des_lengths)
#     indices_test_relation_lengths=pythonList_into_theanoIntMatrix(test_relation_lengths)
#     indices_test_mention_char_ids=pythonList_into_theanoIntMatrix(test_mention_char_ids)
#     indices_test_remainQ_word_ids=pythonList_into_theanoIntMatrix(test_remainQ_word_ids)
#     indices_test_mention_char_lens=pythonList_into_theanoIntMatrix(test_mention_char_lens)
#     indices_test_remainQ_word_len=pythonList_into_theanoIntMatrix(test_remainQ_word_len)   
#     indices_test_entity_scores=pythonList_into_theanoIntMatrix(test_entity_scores)

    rand_values=random_value_normal((vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
#     rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
    #rand_values[0]=numpy.array([1e-50]*emb_size)
#     rand_values=load_word2vec_to_init(rand_values, rootPath+'word_emb.txt')
    embeddings=theano.shared(value=rand_values, borrow=True)      

    char_rand_values=random_value_normal((char_size+1, char_emb_size), theano.config.floatX, numpy.random.RandomState(1234))
#     char_rand_values[0]=numpy.array(numpy.zeros(char_emb_size),dtype=theano.config.floatX)
    char_embeddings=theano.shared(value=char_rand_values, borrow=True)      

    
    # allocate symbolic variables for the data
    index = T.iscalar()
    chosed_indices=T.ivector()
    
    ent_char_ids_M = T.imatrix()   
    ent_lens_M = T.imatrix()
    men_char_ids_M = T.imatrix()  
    men_lens_M=T.imatrix()
    rel_word_ids_M=T.imatrix()
    rel_word_lens_M=T.imatrix()
    desH_word_ids_M=T.imatrix()
    desH_word_lens_M=T.imatrix()
    q_word_ids_M=T.imatrix()
    q_word_lens_M=T.imatrix()
    ent_scores=T.fvector()

    
    filter_size=(emb_size,window_width)
    char_filter_size=(char_emb_size, window_width)
    #poolsize1=(1, ishape[1]-filter_size[1]+1) #?????????????????????????????
#     length_after_wideConv=ishape[1]+filter_size[1]-1
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    

    char_filter_shape=(char_nkerns, 1, char_filter_size[0], char_filter_size[1])
    word_filter_shape=(word_nkerns, 1, filter_size[0], filter_size[1])
    char_conv_W, char_conv_b=create_conv_para(rng, filter_shape=char_filter_shape)
    q_rel_conv_W, q_rel_conv_b=create_conv_para(rng, filter_shape=word_filter_shape)
    q_desH_conv_W, q_desH_conv_b=create_conv_para(rng, filter_shape=word_filter_shape)
    params = [char_embeddings, embeddings, char_conv_W, char_conv_b, q_rel_conv_W, q_rel_conv_b, q_desH_conv_W, q_desH_conv_b]
    load_model_from_file(rootPath, params, mark)

    def SimpleQ_matches_Triple(ent_char_ids_f,ent_lens_f,rel_word_ids_f,rel_word_lens_f,desH_word_ids_f,
                       desH_word_lens_f,
                       men_char_ids_f, q_word_ids_f, men_lens_f, q_word_lens_f):
        

#         rng = numpy.random.RandomState(23455)
        ent_char_input = char_embeddings[ent_char_ids_f.flatten()].reshape((batch_size,max_char_len, char_emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
        men_char_input = char_embeddings[men_char_ids_f.flatten()].reshape((batch_size,max_char_len, char_emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
        
        rel_word_input = embeddings[rel_word_ids_f.flatten()].reshape((batch_size,max_relation_len, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
        desH_word_input = embeddings[desH_word_ids_f.flatten()].reshape((batch_size,max_des_len, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
        
#         desT_word_input = embeddings[desT_word_ids_f.flatten()].reshape((batch_size,max_des_len, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
        q_word_input = embeddings[q_word_ids_f.flatten()].reshape((batch_size,max_Q_len, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    
    
        #ent_mention
        ent_char_conv = Conv_with_input_para(rng, input=ent_char_input,
                image_shape=(batch_size, 1, char_emb_size, max_char_len),
                filter_shape=char_filter_shape, W=char_conv_W, b=char_conv_b)
        men_char_conv = Conv_with_input_para(rng, input=men_char_input,
                image_shape=(batch_size, 1, char_emb_size, max_char_len),
                filter_shape=char_filter_shape, W=char_conv_W, b=char_conv_b)
        #q-rel
        q_rel_conv = Conv_with_input_para(rng, input=q_word_input,
                image_shape=(batch_size, 1, emb_size, max_Q_len),
                filter_shape=word_filter_shape, W=q_rel_conv_W, b=q_rel_conv_b)
        rel_conv = Conv_with_input_para(rng, input=rel_word_input,
                image_shape=(batch_size, 1, emb_size, max_relation_len),
                filter_shape=word_filter_shape, W=q_rel_conv_W, b=q_rel_conv_b)
        #q_desH
        q_desH_conv = Conv_with_input_para(rng, input=q_word_input,
                image_shape=(batch_size, 1, emb_size, max_Q_len),
                filter_shape=word_filter_shape, W=q_desH_conv_W, b=q_desH_conv_b)
        desH_conv = Conv_with_input_para(rng, input=desH_word_input,
                image_shape=(batch_size, 1, emb_size, max_des_len),
                filter_shape=word_filter_shape, W=q_desH_conv_W, b=q_desH_conv_b)
        
        ent_conv_pool=Max_Pooling(rng, input_l=ent_char_conv.output, left_l=ent_lens_f[0], right_l=ent_lens_f[2])
        men_conv_pool=Max_Pooling(rng, input_l=men_char_conv.output, left_l=men_lens_f[0], right_l=men_lens_f[2])
        
        #q_rel_pool=Max_Pooling(rng, input_l=q_rel_conv.output, left_l=q_word_lens_f[0], right_l=q_word_lens_f[2])
        rel_conv_pool=Max_Pooling(rng, input_l=rel_conv.output, left_l=rel_word_lens_f[0], right_l=rel_word_lens_f[2])
        q_rel_pool=Average_Pooling_for_SimpleQA(rng, input_l=q_rel_conv.output, input_r=rel_conv_pool.output_maxpooling,
                                                left_l=q_word_lens_f[0], right_l=q_word_lens_f[2], length_l=q_word_lens_f[1]+filter_size[1]-1,
                                                dim=max_Q_len+filter_size[1]-1, topk=2)
        

        q_desH_pool=Max_Pooling(rng, input_l=q_desH_conv.output, left_l=q_word_lens_f[0], right_l=q_word_lens_f[2])
        desH_conv_pool=Max_Pooling(rng, input_l=desH_conv.output, left_l=desH_word_lens_f[0], right_l=desH_word_lens_f[2])
        

        overall_simi=cosine(ent_conv_pool.output_maxpooling, men_conv_pool.output_maxpooling)*0.33333+\
                    cosine(q_rel_pool.output_maxpooling, rel_conv_pool.output_maxpooling)*0.45+\
                    0.0*cosine(q_desH_pool.output_maxpooling, desH_conv_pool.output_maxpooling)
#                     cosine(q_desT_pool.output_maxpooling, desT_conv_pool.output_maxpooling)
        return overall_simi
    
    simi_list, updates = theano.scan(
        SimpleQ_matches_Triple,
                sequences=[ent_char_ids_M,ent_lens_M,rel_word_ids_M,rel_word_lens_M,desH_word_ids_M,
                   desH_word_lens_M,
                   men_char_ids_M, q_word_ids_M, men_lens_M, q_word_lens_M])
    
    simi_list+=2.6*ent_scores
    
    posi_simi=simi_list[0]
    nega_simies=simi_list[1:]
    loss_simi_list=T.maximum(0.0, margin-posi_simi.reshape((1,1))+nega_simies) 
    loss_simi=T.sum(loss_simi_list)

    




    test_model = theano.function([ent_char_ids_M, ent_lens_M, men_char_ids_M, men_lens_M, rel_word_ids_M, rel_word_lens_M, desH_word_ids_M, desH_word_lens_M,
                                  q_word_ids_M, q_word_lens_M, ent_scores], [loss_simi, simi_list],on_unused_input='ignore')



    ###############
    # TRAIN MODEL #
    ###############
    print '... testing'

    start_time = time.clock()
    mid_time = start_time

    epoch = 0



                 
    test_loss=[]
    succ=0
    for i in range(test_size):
        
        #prepare data
        test_ent_char_ids_M= numpy.asarray(test_pos_entity_char[i], dtype='int32').reshape((length_per_example_test[i], max_char_len))  
        test_ent_lens_M = numpy.asarray(test_entity_char_lengths[i], dtype='int32').reshape((length_per_example_test[i], 3))
        test_men_char_ids_M = numpy.asarray(test_mention_char_ids[i], dtype='int32').reshape((length_per_example_test[i], max_char_len))
        test_men_lens_M = numpy.asarray(test_mention_char_lens[i], dtype='int32').reshape((length_per_example_test[i], 3))
        test_rel_word_ids_M = numpy.asarray(test_relations[i], dtype='int32').reshape((length_per_example_test[i], max_relation_len))  
        test_rel_word_lens_M = numpy.asarray(test_relation_lengths[i], dtype='int32').reshape((length_per_example_test[i], 3))
        test_desH_word_ids_M =numpy.asarray( test_pos_entity_des[i], dtype='int32').reshape((length_per_example_test[i], max_des_len))
        test_desH_word_lens_M = numpy.asarray(test_entity_des_lengths[i], dtype='int32').reshape((length_per_example_test[i], 3))
        test_q_word_ids_M = numpy.asarray(test_remainQ_word_ids[i], dtype='int32').reshape((length_per_example_test[i], max_Q_len))
        test_q_word_lens_M = numpy.asarray(test_remainQ_word_len[i], dtype='int32').reshape((length_per_example_test[i], 3))
        test_ent_scores = numpy.asarray(test_entity_scores[i], dtype=theano.config.floatX)
    
    
    
    
                    
        loss_simi_i,simi_list_i=test_model(test_ent_char_ids_M, test_ent_lens_M, test_men_char_ids_M, test_men_lens_M, test_rel_word_ids_M, test_rel_word_lens_M,
                                           test_desH_word_ids_M, test_desH_word_lens_M, test_q_word_ids_M, test_q_word_lens_M, test_ent_scores)
    #                     print 'simi_list_i:', simi_list_i[:10]
        test_loss.append(loss_simi_i)
        if len(simi_list_i)==1 or simi_list_i[0]>=max(simi_list_i[1:]):
            succ+=1
        if i%1000==0:
            print 'testing', i, '...acc:', (succ*1.0/(i+1))*(19835*1.0/21687)
    succ=succ*100.0/21687
    #now, check MAP and MRR
    print 'accu:', succ
    

#     store_model_to_file(rootPath, params, succ, mark)

    print 'Epoch ', epoch, 'uses ', (time.clock()-mid_time)/60.0, 'min'



def store_model_to_file(path, best_params, acc, mark):
    save_file = open(path+'Best_Conv_Para'+mark+str(acc)[:5], 'wb')  # this will overwrite current contents
    for para in best_params:           
        cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()
    print 'Better model stored'

def load_model_from_file(path, params, acc):
    save_file = open(path+'Best_Conv_Para'+str(acc))
    for para in params:
        para.set_value(cPickle.load(save_file), borrow=True)
    save_file.close()   
    print 'model initialized over'
def Linear(sum_uni_l, sum_uni_r):
    return (T.dot(sum_uni_l,sum_uni_r.T)).reshape((1,1))    
def Poly(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    poly=(0.5*dot+1)**3
    return poly.reshape((1,1))    
def Sigmoid(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    return T.tanh(1.0*dot+1).reshape((1,1))    
def RBF(sum_uni_l, sum_uni_r):
    eucli=T.sum((sum_uni_l-sum_uni_r)**2)
    return T.exp(-0.5*eucli).reshape((1,1))    
def GESD (sum_uni_l, sum_uni_r):
    eucli=1/(1+T.sum((sum_uni_l-sum_uni_r)**2))
    kernel=1/(1+T.exp(-(T.dot(sum_uni_l,sum_uni_r.T)+1)))
    return (eucli*kernel).reshape((1,1))   
def EUCLID(sum_uni_l, sum_uni_r):
    return T.sqrt(T.sqr(sum_uni_l-sum_uni_r).sum()+1e-20).reshape((1,1))
    


if __name__ == '__main__':
    evaluate_lenet5()
