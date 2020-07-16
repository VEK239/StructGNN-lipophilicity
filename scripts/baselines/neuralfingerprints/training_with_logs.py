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



parser = ArgumentParser()
parser.add_argument("-f", "--fp_length", dest="fp_length",
                    help="length of fingerprint",default = 50,type=int)
parser.add_argument("-d", "--fp_depth",
                    dest="fp_depth", default=4,
                    help="depth of fingerprint (radius)",type=int)
parser.add_argument("-i", "--init_scale",
                    dest="init_scale", default=-4,
                    help="power of exponent for scale of weights initialization",type=float)
parser.add_argument("-l", "--learn_rate",
                    dest="learn_rate", default=-3,
                    help="power of exponent for learning rate",type=float)
parser.add_argument("-s", "--l2_penalty",
                    dest="l2_penalty", default=-2,
                    help="power of exponent for size of l2 penalty", type=float)
parser.add_argument("-n", "--num_exp",
                    dest="NUM_EXP", default='',
                    help="number of current experiment")
parser.add_argument("-p", "--l1_penalty",
                    dest="l1_penalty", default=0.0,
                    help="power of exponent for size of l1 penalty", type=float)
parser.add_argument("-a", "--h1_size",
                    dest="h1_size", default=100,
                    help="Size of layer after fingerprint",type=int)
parser.add_argument("-c", "--conv_width",
                    dest="conv_width", default=20,
                    help="size of convolutions",type=int)
parser.add_argument("-t", "--data_file",
                    dest="data_file", default='logp_mean',
                    help="Prefix of filename with data",type=str)
parser.add_argument("-e", "--num_epochs",
                    dest="num_epochs", default=10,
                    help="Number of epochs",type=int)
parser.add_argument("-b", "--early_stopping",
                    dest="early_stopping", default=500,
                    help="Number of iterations before early stopping",type=int)

args = parser.parse_args()

# path to datasets
DATASET_PATH = "../../../data/3_final_data/split_data"

# path to logs directory
EXPERIMENTS_DATA = "../../../data/raw/baselines/neuralfingerprint"

# logs path
global LOG_PATH
LOG_PATH=os.path.join(EXPERIMENTS_DATA, "logs")

Path(LOG_PATH).mkdir(exist_ok=True)

global MODEL_PATH
MODEL_PATH=os.path.join(EXPERIMENTS_DATA, "models")

Path(MODEL_PATH).mkdir(exist_ok=True)

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

def main():
    global params
    global task_params
    global vanilla_net_params
    global train_inputs, train_targets
    global val_inputs,   val_targets
    global test_inputs,  test_targets
    global LOG_PATH
    global MODEL_PATH
    
    path = os.path.join(LOG_PATH,'exp_'+args.NUM_EXP)
    Path(path).mkdir(exist_ok=True)
    LOG_PATH = path
    
    path = os.path.join(MODEL_PATH,'exp_'+args.NUM_EXP)
    Path(path).mkdir(exist_ok=True)
    MODEL_PATH = path

    # create log files
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_parameters.json'),'w') as f:
        json.dump(vars(args), f)
    f_log=open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'w')
    f_log.close()
#     start_time=time.time()
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs_metrics_morgan.txt'),'w') as f_log_metrics:
        f_log_metrics.write('train RMSE\tval RMSE\ttrain R2\tval R2\n')
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs_metrics_conv.txt'),'w') as f_log_metrics:
        f_log_metrics.write('train RMSE\tval RMSE\ttrain R2\tval R2\n')
        
    # set data parameters
    task_params = {'target_name' : 'logP',
               'data_file'   : args.data_file}
    
    # load data
    data = load_data(dataset_path=DATASET_PATH, prefix_name = task_params['data_file'], VALUE_COLUMN = task_params['target_name'])

    train_inputs, train_targets = data['train']
    val_inputs,   val_targets   = data['validation']
    test_inputs,  test_targets  = data['test']

    # set parameters of training    
    params = {'fp_length': args.fp_length,
            'fp_depth': args.fp_depth,
            'init_scale': np.exp(args.init_scale),
            'learn_rate': np.exp(args.learn_rate),
            'l2_penalty': np.exp(args.l2_penalty),
            'l1_penalty': 0,
              'h1_size':args.h1_size,
            'conv_width':args.conv_width,
             'batch_size':100}
    
    params['num_iters'] = args.num_epochs*len(train_inputs)/params['batch_size']
    
    # create predictive model (regressor)
    vanilla_net_params = dict(
    layer_sizes = [params['fp_length'], params['h1_size']],  # One hidden layer.
    normalize=True, L2_reg = params['l2_penalty'], L1_reg = params['l1_penalty'], nll_func = rmse)
    
    
    # start training
    print "Task params", task_params, params
    print
    print "Starting Morgan fingerprint experiment..."
    write_log_file(LOG_PATH, args.NUM_EXP+'_logs.txt', "\nStarting Morgan fingerprint experiment..."+'\n')
    test_loss_morgan = run_morgan_experiment()
    print "Starting Average experiment..."
    write_log_file(LOG_PATH, args.NUM_EXP+'_logs.txt', "\nStarting Average experiment..."+'\n')
    test_loss_avg = run_avg_experiment()
    print "Starting neural fingerprint experiment..."
    write_log_file(LOG_PATH, args.NUM_EXP+'_logs.txt', "\nStarting neural fingerprint experiment..."+'\n')
    test_rmse_neural, test_r2_neural = run_conv_experiment()
    print
    print "\nMorgan test R2:", test_loss_morgan, "\nAvg test R2:", test_loss_avg, "\nNeural test R2:", test_r2_neural
    write_log_file(LOG_PATH, args.NUM_EXP+'_logs.txt', "\nMorgan test R2: "+\
                   str(test_loss_morgan) +
                   "\nAvg test R2: "+ str(test_loss_avg) + 
                   "\nNeural test R2: " + str(test_r2_neural) +
                   "\nNeural test RMSE: "+str(test_rmse_neural)+'\n')


if __name__ == '__main__':
    main()
