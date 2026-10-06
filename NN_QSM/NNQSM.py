## Import dependent libraries
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Concatenate
from keras import optimizers, callbacks
from datetime import datetime, timedelta

## Import libraries developed by this study
from hydrolayer import PRNNLayer
from hydrodata import DataforIndividual
import hydroutils

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'

from multiprocessing import Pool



def generate_train_test(train_set, test_set, wrap_length):
    train_x_np = train_set.values[:, :-1]
    train_y_np = train_set.values[:, -1:]
    test_x_np = test_set.values[:, :-1]
    test_y_np = test_set.values[:, -1:]
    
    wrap_number_train = (train_set.shape[0]-wrap_length)//365 + 1
    
    train_x = np.empty(shape = (wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty(shape = (wrap_number_train, wrap_length, train_y_np.shape[1]))

    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)
    
    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i*365:(wrap_length+i*365), :]
        train_y[i, :, :] = train_y_np[i*365:(wrap_length+i*365), :]
             
    return train_x, train_y, test_x, test_y


def create_model(input_shape, seed, num_filters, model_type='hybrid'):
    """Create a Keras model.
    -- input_shape: the shape of input, controlling the time sequence length of the P-RNN
    -- seed: the random seed for the weights initialization of the 1D-CNN layers
    -- num_filters: the number of filters for the 1D-CNN layer
    -- model_type: can be 'hybrid', 'physical', or 'common'
    """
    
    x_input = Input(shape=input_shape, name='Input') 
    
    if model_type == 'physical':
        hydro_output = PRNNLayer(seed = seed, mode= 'normal', name='Hydro')(x_input)
        model = Model(x_input, hydro_output)
        
                
    else:
        hydro_output = PRNNLayer(seed = seed, mode= 'analysis', name='Hydro')(x_input)
        model = Model(x_input, hydro_output)   
 
    return model

def train_model(model, train_x, train_y, ep_number, lrate, save_path):
    """Train a Keras model.
    -- model: the Keras model object
    -- train_x, train_y: the input and target for training the model
    -- ep_number: the maximum epoch number
    -- lrate: the initial learning rate
    -- save_path: where the model will be saved
    """
    
    save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                     save_weights_only=True)
    es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                 restore_best_weights=True)
    reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                         min_delta=0.005, cooldown=0, min_lr=lrate / 100)
    tnan = callbacks.TerminateOnNaN()

    model.compile(loss=hydroutils.nse_loss, metrics=[hydroutils.nse_metrics], optimizer=optimizers.Adam(lr=lrate))
    history = model.fit(train_x, train_y, epochs=ep_number, batch_size=10000, callbacks=[save, es, reduce, tnan])
    
    return history
    
def test_model(model, test_x, save_path):
    """Test a Keras model.
    -- model: the Keras model object
    -- test_x: the input for testing the model
    -- save_path: where the model was be saved
    """
    model.load_weights(save_path, by_name=True)
    pred_y = model.predict(test_x, batch_size=10000)
    
    return pred_y

def addYears(date, years):
    result = date + timedelta(366 * years)
    if years > 0:
        while result.year - date.year > years or date.month < result.month or date.day < result.day:
            result += timedelta(-1)
    elif years < 0:
        while result.year - date.year < years or date.month > result.month or date.day > result.day:
            result += timedelta(1)
    return result

    

def models(i, m):
    working_path = '/path/to/CAMELS/DATA/'
    results_path = '/path/to/model_outputs/'
    basin_list = pd.read_csv(os.path.join(working_path, 'basin_list_cold.txt'),sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str})
    
    training_start = '1980-10-01'
    training_end= '2000-09-30'  
    testing_start = '1999-10-01' 
    testing_end= '2010-09-30'
    wrap_length=2190 # It can be other values, but recommend this value should not be less than 5 years (1825 days).    
    
    evaluation_start = datetime.strftime(addYears(datetime.strptime(testing_start, '%Y-%m-%d'), 1), '%Y-%m-%d')
    evaluation_end = testing_end
    date_range = pd.date_range(evaluation_start, evaluation_end)     

    basin_id = basin_list["BASIN_ID"][i] # The basin_id can be changed to any 8-digit basin id contained in the basin_list.txt
    hydrodata = DataforIndividual(working_path, basin_id).load_data()
    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]
    train_x, train_y, test_x, test_y = generate_train_test(train_set, test_set, wrap_length=wrap_length)
    Path(f'{results_path}/NNQSM/MODEL/NUM{m+1}').mkdir(parents=True, exist_ok=True)
    Path(f'{results_path}/NNQSM/RESULTS/NUM{m+1}').mkdir(parents=True, exist_ok=True)
    save_path_physical = f'{results_path}/NNQSM/MODEL/NUM{m+1}/{basin_id}_qsm.h5'
    
    model = create_model((train_x.shape[1], train_x.shape[2]), seed = m*100, num_filters = 8, model_type='physical')
    model.summary()
    hybrid_history = train_model(model, train_x, train_y, ep_number=600, lrate=0.01, save_path=save_path_physical)
    
    model = create_model((test_x.shape[1], test_x.shape[2]), seed = m*100, num_filters = 8, model_type='analysis')
    flow = test_model(model, test_x, save_path_physical)
    evaluate_set = test_set.loc[:, ['prcp(mm/day)','flow(mm)']]
    evaluate_set['flow_obs'] = evaluate_set['flow(mm)']
    variables = ['ps', 'pr', 's0', 's1', 'm', 'et','flow_sim']
    
    print (np.shape(flow))
    
    for tt in range (7):
        
        evaluate_set[variables[tt]] = np.clip(flow[0, :, tt], a_min = 0, a_max = None)
    
    # We only evaluate the data in the evaluation period
    
    evaluate_set = evaluate_set[evaluate_set.index.isin(date_range)]
    
    evaluate_set.to_excel(f'{results_path}/NNQSM/RESULTS/NUM{m+1}/{basin_id}_qsm.xlsx')
    
    return ()


        
if __name__ == '__main__':
    

    for m in range(50):
        
        i_values = np.linspace(0, 30, 31)  
        
        params = [(i, m) for i in i_values]
        
        with Pool(8) as p:
            
            p.starmap(models, params)    
        




