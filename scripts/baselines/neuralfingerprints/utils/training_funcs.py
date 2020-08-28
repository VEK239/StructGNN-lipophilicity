import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

# Path to Neural Fingerprint scripts
import sys
sys.path
sys.path.append('./utils')

from build_vanilla_net import build_morgan_deep_net
from build_convnet import build_conv_deep_net
from util import normalize_array, build_batched_grad
from optimizers import adam
from util import rmse
from sklearn.metrics import r2_score

from pathlib2 import Path
import time
import os
import json
import pickle
from argparse import ArgumentParser





def write_log_file(path, filename, log_message):
    """Write info to txt files"""
    with open(os.path.join(path, filename),'a') as f_log:
        f_log.write(log_message)
        
def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, params, seed=0,
             validation_smiles=None, validation_raw_targets=None, filename_fix=''):
    """Function to train model based on pred_fun and loss_fun"""
    write_log_file(LOG_PATH, args.NUM_EXP+'_logs.txt', "\nTotal number of weights in the network: "+str(num_weights)+'\n')
    print ("Total number of weights in the network:", num_weights)
    init_weights = npr.RandomState(seed).randn(num_weights) * params['init_scale']

    num_print_examples = len(train_smiles)
    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = []
    def callback(weights, iter):
        if iter % 10 == 0:
            print ("max of weights", np.max(np.abs(weights)))
            write_log_file(LOG_PATH, args.NUM_EXP+'_logs.txt', "\nmax of weights "+str(np.max(np.abs(weights)))+'\n')
            train_preds = undo_norm(pred_fun(weights, train_smiles[:num_print_examples]))
            cur_loss = loss_fun(weights, train_smiles[:num_print_examples], train_targets[:num_print_examples])
            training_curve.append(cur_loss)
            print ("Iteration", iter, "loss", cur_loss,\
                  "train RMSE", rmse(train_preds, train_raw_targets[:num_print_examples])),
            print "Train R2", iter, ":", \
                    r2_score(train_raw_targets, train_preds)
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print ("Validation RMSE", iter, ":", rmse(validation_preds, validation_raw_targets)),
                print "Validation R2", iter, ":", \
                    r2_score(validation_raw_targets, validation_preds),
            write_log_file(LOG_PATH, args.NUM_EXP+'_logs_metrics_'+filename_fix+'.txt',\
                           str(rmse(train_preds, train_raw_targets[:num_print_examples]))+'\t'+\
                           str(rmse(validation_preds, validation_raw_targets))+'\t'+\
                           str(r2_score(train_raw_targets, train_preds))+'\t'+\
                           str(r2_score(validation_raw_targets, validation_preds))+'\t'+\
                           '\n')
            if filename_fix=='conv':
                with open(os.path.join(MODEL_PATH,'model_'+str(iter)+'.pkl'),'w') as f:
                    pickle.dump(weights, f)
            return rmse(validation_raw_targets, validation_preds)
    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    trained_weights, best_iter = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=params['num_iters'], step_size=params['learn_rate'], delta=args.early_stopping)
    print "Best model is ", best_iter
    if filename_fix=='conv':
        with open(os.path.join(MODEL_PATH,'model_best'+str(best_iter)+'.pkl'),'w') as f:
            pickle.dump(trained_weights, f)

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve

def print_performance(pred_func):
    """Print and log quality metrics"""
    train_preds = pred_func(train_inputs)
    test_preds = pred_func(test_inputs)
    print "\nPerformance on " + task_params['target_name'] + ":"
    print "\nTrain RMSE:", rmse(train_preds, train_targets)
    print "\nTrain R2:", r2_score(train_targets, train_preds)
    print "\nTest RMSE: ", rmse(test_preds,  test_targets)
    print "\nTest R2: ", r2_score(test_targets, test_preds)
    print "-" * 80
    write_log_file(LOG_PATH, args.NUM_EXP+'_logs.txt', \
                   "\nPerformance on " + task_params['target_name'] + ":"+\
                   "\nTrain RMSE: "+str(rmse(train_preds, train_targets))+\
                   "\nTrain R2: "+str(r2_score(train_targets, train_preds))+\
                   "\nTest RMSE: "+str(rmse(test_preds,  test_targets))+\
                   "\nTest R2: "+str(r2_score(test_targets, test_preds))+
                   '\n')
    return r2_score(test_targets, test_preds)

def run_morgan_experiment():
    global params
    loss_fun, pred_fun, net_parser = \
        build_morgan_deep_net(params['fp_length'],
                              params['fp_depth'], vanilla_net_params)
    num_weights = len(net_parser)
    predict_func, trained_weights, conv_training_curve = \
        train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                 params, validation_smiles=val_inputs, validation_raw_targets=val_targets, filename_fix='morgan')
    return print_performance(predict_func)

def run_conv_experiment():
    conv_layer_sizes = [params['conv_width']] * params['fp_depth']
    conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                        'fp_length' : params['fp_length'], 'normalize' : 1}
    loss_fun, pred_fun, conv_parser = \
        build_conv_deep_net(conv_arch_params, vanilla_net_params, params['l2_penalty'])
    num_weights = len(conv_parser)
    predict_func, trained_weights, conv_training_curve = \
        train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                 params, validation_smiles=val_inputs, validation_raw_targets=val_targets, filename_fix='conv')
    test_predictions = predict_func(test_inputs)
    return rmse(test_targets, test_predictions), r2_score(test_targets, test_predictions)

def run_avg_experiment():
    y_train_mean = np.mean(train_targets)
    return print_performance(lambda x : y_train_mean*np.ones(len(x)))


def load_data(dataset_path, prefix_name, VALUE_COLUMN = 'logP', SMILES_COLUMN='smiles'):
    """load datasets, prefix_name - filename without csv format"""
    import pandas as pd
    import os
    
    data_splits = ['train', 'test', 'validation']
    
    datasets = {}
    
    for split in data_splits:
        data = pd.read_csv(os.path.join(dataset_path,prefix_name+'_'+split+'.csv'))
        datasets[split] = (data[SMILES_COLUMN].values, data[VALUE_COLUMN].values)
        
    
    return datasets