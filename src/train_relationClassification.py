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
from load_data_relationClassification import load_train#, load_word2vec_to_init#, load_mts_wikiQA, load_wmf_wikiQA
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Conv_with_input_para, Average_Pooling_for_SimpleQA, create_conv_para, pythonList_into_theanoIntMatrix, Max_Pooling, cosine, pythonList_into_theanoFloatMatrix, Diversify_Reg
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

def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, word_nkerns=300, batch_size=1, window_width=[3,3],
                    emb_size=300, 
                    margin=0.5, L2_weight=0.0003, Div_reg=0.03, update_freq=1, norm_threshold=5.0, max_truncate=40, 
                    max_relation_len=6, max_Q_len=30, 
                    neg_all=100, train_size=69967, test_size=19953, mark='_RC_newdata'):  #train_size=75909, test_size=17386
#     maxSentLength=max_truncate+2*(window_width-1)
    model_options = locals().copy()
    print "model options", model_options
    rootPath='/home/wyin/Datasets/SimpleQuestions_v2/relation_classification/'
    triple_files=['train.replace_ne.withpoolwenpengFormat.txt', 'test.replace_ne.withpoolwenpengFormat.txt']

    rng = numpy.random.RandomState(23455)
    datasets, datasets_test, length_per_example_train, length_per_example_test, vocab_size=load_train(triple_files[0], triple_files[1], max_relation_len, max_Q_len, train_size, test_size, mark)#max_char_len, max_des_len, max_relation_len, max_Q_len

    
    print 'vocab_size:', vocab_size

    train_data=datasets
#     valid_data=datasets[1]
    test_data=datasets_test
#     result=(pos_entity_char, pos_entity_des, relations, entity_char_lengths, entity_des_lengths, relation_lengths, mention_char_ids, remainQ_word_ids, mention_char_lens, remainQ_word_lens, entity_scores)
#     

    train_relations=train_data[0]
    train_relation_lengths=train_data[1]
    train_remainQ_word_ids=train_data[2]
    train_remainQ_word_len=train_data[3]

    test_relations=test_data[0]
    test_relation_lengths=test_data[1]
    test_remainQ_word_ids=test_data[2]
    test_remainQ_word_len=test_data[3]


    

    train_sizes=[len(train_relations),len(train_relation_lengths),len(train_remainQ_word_ids), len(train_remainQ_word_len)]
    if sum(train_sizes)/len(train_sizes)!=train_size:
        print 'weird size:', train_sizes
        exit(0)

    test_sizes=[len(test_relations),len(test_relation_lengths),  len(test_remainQ_word_ids),len(test_remainQ_word_len)]
    if sum(test_sizes)/len(test_sizes)!=test_size:
        print 'weird size:', test_sizes
        exit(0)

    n_train_batches=train_size/batch_size
    n_test_batches=test_size/batch_size
    

    
    
#     indices_train_pos_entity_char=theano.shared(numpy.asarray(train_pos_entity_char, dtype='int32'), borrow=True)
#     indices_train_pos_entity_des=theano.shared(numpy.asarray(train_pos_entity_des, dtype='int32'), borrow=True)
#     indices_train_relations=theano.shared(numpy.asarray(train_relations, dtype='int32'), borrow=True)
#     indices_train_entity_char_lengths=theano.shared(numpy.asarray(train_entity_char_lengths, dtype='int32'), borrow=True)
#     indices_train_entity_des_lengths=theano.shared(numpy.asarray(train_entity_des_lengths, dtype='int32'), borrow=True)
#     indices_train_relation_lengths=theano.shared(numpy.asarray(train_relation_lengths, dtype='int32'), borrow=True)
#     indices_train_mention_char_ids=theano.shared(numpy.asarray(train_mention_char_ids, dtype='int32'), borrow=True)
#     indices_train_remainQ_word_ids=theano.shared(numpy.asarray(train_remainQ_word_ids, dtype='int32'), borrow=True)
#     indices_train_mention_char_lens=theano.shared(numpy.asarray(train_mention_char_lens, dtype='int32'), borrow=True)
#     indices_train_remainQ_word_len=theano.shared(numpy.asarray(train_remainQ_word_len, dtype='int32'), borrow=True)
#     indices_train_entity_scores=theano.shared(numpy.asarray(train_entity_scores, dtype=theano.config.floatX), borrow=True)
    


    rand_values=random_value_normal((vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
    #rand_values[0]=numpy.array([1e-50]*emb_size)
#     rand_values=load_word2vec_to_init(rand_values, rootPath+'word_emb'+mark+'.txt')
    embeddings=theano.shared(value=rand_values, borrow=True)      
    

    
    # allocate symbolic variables for the data
    index = T.iscalar()
    rel_word_ids_M=T.imatrix()
    rel_word_lens_M=T.imatrix()
    q_word_ids_f=T.ivector()
    q_word_lens_f=T.ivector()

    
    filter_size=(emb_size,window_width[0])
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    word_filter_shape=(word_nkerns, 1, filter_size[0], filter_size[1])
    q_rel_conv_W, q_rel_conv_b=create_conv_para(rng, filter_shape=word_filter_shape)
    params = [embeddings,q_rel_conv_W, q_rel_conv_b]
    q_rel_conv_W_into_matrix=q_rel_conv_W.reshape((q_rel_conv_W.shape[0], q_rel_conv_W.shape[2]*q_rel_conv_W.shape[3]))
#     load_model_from_file(rootPath, params, '')

    def SimpleQ_matches_Triple(rel_word_ids_f,rel_word_lens_f):
        rel_word_input = embeddings[rel_word_ids_f.flatten()].reshape((batch_size,max_relation_len, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
        q_word_input = embeddings[q_word_ids_f.flatten()].reshape((batch_size,max_Q_len, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)

        #q-rel
        q_rel_conv = Conv_with_input_para(rng, input=q_word_input,
                image_shape=(batch_size, 1, emb_size, max_Q_len),
                filter_shape=word_filter_shape, W=q_rel_conv_W, b=q_rel_conv_b)
        rel_conv = Conv_with_input_para(rng, input=rel_word_input,
                image_shape=(batch_size, 1, emb_size, max_relation_len),
                filter_shape=word_filter_shape, W=q_rel_conv_W, b=q_rel_conv_b)

        
#         q_rel_pool=Max_Pooling(rng, input_l=q_rel_conv.output, left_l=q_word_lens_f[0], right_l=q_word_lens_f[2])
        rel_conv_pool=Max_Pooling(rng, input_l=rel_conv.output, left_l=rel_word_lens_f[0], right_l=rel_word_lens_f[2])
        q_rel_pool=Average_Pooling_for_SimpleQA(rng, input_l=q_rel_conv.output, input_r=rel_conv_pool.output_maxpooling, 
                                                left_l=q_word_lens_f[0], right_l=q_word_lens_f[2], length_l=q_word_lens_f[1]+filter_size[1]-1, 
                                                dim=max_Q_len+filter_size[1]-1, topk=2)
        
   
        
        
        overall_simi=cosine(q_rel_pool.output_maxpooling, rel_conv_pool.output_maxpooling)
        return overall_simi
    
    simi_list, updates = theano.scan(
        SimpleQ_matches_Triple,
                sequences=[rel_word_ids_M,rel_word_lens_M])

    
    posi_simi=simi_list[0]
    nega_simies=simi_list[1:]
    loss_simi_list=T.maximum(0.0, margin-posi_simi.reshape((1,1))+nega_simies) 
    loss_simi=T.sum(loss_simi_list)

    

    
    #L2_reg =(layer3.W** 2).sum()+(layer2.W** 2).sum()+(layer1.W** 2).sum()+(conv_W** 2).sum()
    L2_reg =debug_print((embeddings** 2).sum()+(q_rel_conv_W** 2).sum(), 'L2_reg')#+(layer1.W** 2).sum()++(embeddings**2).sum()
    diversify_reg= Diversify_Reg(q_rel_conv_W_into_matrix)
    cost=loss_simi+L2_weight*L2_reg+Div_reg*diversify_reg
    #cost=debug_print((cost_this+cost_tmp)/update_freq, 'cost')
    



    test_model = theano.function([rel_word_ids_M, rel_word_lens_M, q_word_ids_f, q_word_lens_f], [loss_simi, simi_list],on_unused_input='ignore')
    
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        grad_i=debug_print(grad_i,'grad_i')
        acc = acc_i + T.sqr(grad_i)
#         updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc+1e-10)))   #AdaGrad
#         updates.append((acc_i, acc))    
        if param_i == embeddings:
            updates.append((param_i, T.set_subtensor((param_i - learning_rate * grad_i / T.sqrt(acc+1e-10))[0], theano.shared(numpy.zeros(emb_size)))))   #Ada
        else:
            updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc+1e-10)))   #AdaGrad
        updates.append((acc_i, acc)) 

    train_model = theano.function([rel_word_ids_M, rel_word_lens_M, q_word_ids_f, q_word_lens_f], [loss_simi, cost],updates=updates, on_unused_input='ignore')
      
#     train_model = theano.function([index, chosed_indices], [loss_simi, cost], updates=updates,
#           givens={
#             rel_word_ids_M : indices_train_relations[index].reshape((neg_all, max_relation_len))[chosed_indices].reshape((train_neg_size, max_relation_len)),  
#             rel_word_lens_M : indices_train_relation_lengths[index].reshape((neg_all, 3))[chosed_indices].reshape((train_neg_size, 3)),
#             q_word_ids_M : indices_train_remainQ_word_ids[index].reshape((neg_all, max_Q_len))[chosed_indices].reshape((train_neg_size, max_Q_len)), 
#             q_word_lens_M : indices_train_remainQ_word_len[index].reshape((neg_all, 3))[chosed_indices].reshape((train_neg_size, 3))
#             
#             }, on_unused_input='ignore')




    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    mid_time = start_time

    epoch = 0
    done_looping = False
    
    best_test_accu=0.0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0


        for jj in range(train_size): 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + minibatch_index +1
 
            minibatch_index=minibatch_index+1
            #print batch_start
            train_rel_word_ids_M = numpy.asarray(train_relations[jj], dtype='int32').reshape((length_per_example_train[jj], max_relation_len))  
            train_rel_word_lens_M = numpy.asarray(train_relation_lengths[jj], dtype='int32').reshape((length_per_example_train[jj], 3))
            train_q_word_ids_M = numpy.asarray(train_remainQ_word_ids[jj], dtype='int32')#.reshape((length_per_example_train[jj], max_Q_len))
            train_q_word_lens_M = numpy.asarray(train_remainQ_word_len[jj], dtype='int32')#.reshape((length_per_example_train[jj], 3))
            loss_simi_i, cost_i=train_model(train_rel_word_ids_M, train_rel_word_lens_M,train_q_word_ids_M, train_q_word_lens_M)

 
            if iter % n_train_batches == 0:
                print 'training @ iter = '+str(iter)+'\tloss_simi_i: ', loss_simi_i, 'cost_i:', cost_i
            #if iter ==1:
            #    exit(0)
#             
            if iter > 59999 and iter % 10000 == 0:
                 
                test_loss=[]
                succ=0
                for i in range(test_size):
#                     print 'testing', i, '...'
                    #prepare data
                    test_rel_word_ids_M = numpy.asarray(test_relations[i], dtype='int32').reshape((length_per_example_test[i], max_relation_len))  
                    test_rel_word_lens_M = numpy.asarray(test_relation_lengths[i], dtype='int32').reshape((length_per_example_test[i], 3))
                    test_q_word_ids_M = numpy.asarray(test_remainQ_word_ids[i], dtype='int32')#.reshape((length_per_example_test[i], max_Q_len))
                    test_q_word_lens_M = numpy.asarray(test_remainQ_word_len[i], dtype='int32')#.reshape((length_per_example_test[i], 3))
                    loss_simi_i,simi_list_i=test_model(test_rel_word_ids_M, test_rel_word_lens_M,test_q_word_ids_M, test_q_word_lens_M)
#                     print 'simi_list_i:', simi_list_i[:10]
                    test_loss.append(loss_simi_i)
                    if simi_list_i[0]>=max(simi_list_i[1:]):
                        succ+=1
#                     print 'testing', i, '...acc:', succ*1.0/(i+1)
                succ=(succ+20610-test_size)*1.0/20610
                #now, check MAP and MRR
                print(('\t\t\t\t\t\tepoch %i, minibatch %i/%i, test accu of best '
                           'model %f') %
                          (epoch, minibatch_index, n_train_batches,succ))

                if best_test_accu< succ:
                    best_test_accu=succ
                    store_model_to_file(rootPath, params, mark)
            if patience <= iter:
                done_looping = True
                break
        print 'Epoch ', epoch, 'uses ', (time.clock()-mid_time)/60.0, 'min'
        mid_time = time.clock() 

            
        #print 'Batch_size: ', update_freq
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def store_model_to_file(path, best_params, mark):
    save_file = open(path+'Best_Conv_Para'+mark, 'wb')  # this will overwrite current contents
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
