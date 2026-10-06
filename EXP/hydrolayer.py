

from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation
import keras.backend as K
import tensorflow as tf


class PRNNLayer(Layer):
    """Implementation of the standard P-RNN layer
    Hyper-parameters
    ----------
    mode: if in "normal", the output will be the generated flow;
          if in "analysis", the output will be a tensor containing all state variables and process variables
    ==========
    Parameters
    ----------
    f: Rate of decline in flow from catchment bucket | Range: (0, 0.1)
    smax: Maximum storage of the catchment bucket      | Range: (100, 1500)
    qmax: Maximum subsurface flow at full bucket     | Range: (10, 50)
    ddf: Thermal degree‐day factor                     | Range: (0, 5.0)
    tmax: Temperature above which snow starts melting  | Range: (0, 3.0)
    tmin: Temperature below which precipitation is snow| Range: (-3.0, 0)
    """

    def __init__(self, mode='normal', seed = None, **kwargs):
        self.mode = mode
        self.seed = seed
        super(PRNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        param_ranges = {
        'f': {'min': 0.0, 'max': 1.0},      # 融雪因子
        'smax': {'min': 1/15, 'max': 1.0},  # 最大土壤含水量
        'qmax': {'min': 0.2, 'max': 1.0},   # 最大产流量
        'ddf': {'min': 0.0, 'max': 1.0},    # 度日因子
        'tmin': {'min': 0.0, 'max': 1.0},   # 最小温度
        'tmax': {'min': 0.0, 'max': 1.0}    # 最大温度
    }
        
        self.f = self.add_weight(name='f', shape=(1,),
                             initializer=initializers.RandomUniform(
                                 minval=param_ranges['f']['min'], 
                                 maxval=param_ranges['f']['max'], 
                                 seed=self.seed
                             ),
                             constraint=constraints.min_max_norm(
                                 min_value=param_ranges['f']['min'], 
                                 max_value=param_ranges['f']['max'], 
                                 rate=0.9
                             ),
                             trainable=True)
    
        self.smax = self.add_weight(name='smax', shape=(1,),
                                initializer=initializers.RandomUniform(
                                    minval=param_ranges['smax']['min'], 
                                    maxval=param_ranges['smax']['max'], 
                                    seed=self.seed + 5
                                ),
                                constraint=constraints.min_max_norm(
                                    min_value=param_ranges['smax']['min'], 
                                    max_value=param_ranges['smax']['max'], 
                                    rate=0.9
                                ),
                                trainable=True)
    
        self.qmax = self.add_weight(name='qmax', shape=(1,),
                                initializer=initializers.RandomUniform(
                                    minval=param_ranges['qmax']['min'], 
                                    maxval=param_ranges['qmax']['max'], 
                                    seed=self.seed + 10
                                ),
                                constraint=constraints.min_max_norm(
                                    min_value=param_ranges['qmax']['min'], 
                                    max_value=param_ranges['qmax']['max'], 
                                    rate=0.9
                                ),
                                trainable=True)
    
        self.ddf = self.add_weight(name='ddf', shape=(1,),
                               initializer=initializers.RandomUniform(
                                   minval=param_ranges['ddf']['min'], 
                                   maxval=param_ranges['ddf']['max'], 
                                   seed=self.seed - 5
                               ),
                               constraint=constraints.min_max_norm(
                                   min_value=param_ranges['ddf']['min'], 
                                   max_value=param_ranges['ddf']['max'], 
                                   rate=0.9
                               ),
                               trainable=True)
    
        self.tmin = self.add_weight(name='tmin', shape=(1,),
                                initializer=initializers.RandomUniform(
                                    minval=param_ranges['tmin']['min'], 
                                    maxval=param_ranges['tmin']['max'], 
                                    seed=self.seed - 10
                                ),
                                constraint=constraints.min_max_norm(
                                    min_value=param_ranges['tmin']['min'], 
                                    max_value=param_ranges['tmin']['max'], 
                                    rate=0.9
                                ),
                                trainable=True)
    
        self.tmax = self.add_weight(name='tmax', shape=(1,),
                                initializer=initializers.RandomUniform(
                                    minval=param_ranges['tmax']['min'], 
                                    maxval=param_ranges['tmax']['max'], 
                                    seed=self.seed - 15
                        ),
                                constraint=constraints.min_max_norm(
                                    min_value=param_ranges['tmax']['min'], 
                                    max_value=param_ranges['tmax']['max'], 
                                    rate=0.9
                                ),
                                trainable=True)

        super(PRNNLayer, self).build(input_shape)

    def heaviside(self, x):
        """
        A smooth approximation of Heaviside step function
            if x < 0: heaviside(x) ~= 0
            if x > 0: heaviside(x) ~= 1
        """

        return (K.tanh(5 * x) + 1) / 2

    def rainsnowpartition(self, p, t, tmin):
        """
        Equations to partition incoming precipitation into rain or snow
            if t < tmin:
                psnow = p
                prain = 0
            else:
                psnow = 0
                prain = p
        """
        tmin = tmin * -3  # (-3.0, 0)

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]

    def snowbucket(self, s0, t, ddf, tmax):
        """
        Equations for the snow bucket
            if t > tmax:
                if s0 > 0:
                    melt = min(s0, ddf*(t - tmax))
                else:
                    melt = 0
            else:
                melt = 0
        """
        ddf = ddf * 5  # (0, 5.0)
        tmax = tmax * 3  # (0, 3.0)

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt

    def soilbucket(self, s1, pet, f, smax, qmax):
        """
        Equations for the soil bucket
            if s1 < 0:
                et = 0
                qsub = 0
                qsurf = 0
            elif s1 > smax:
                et = pet
                qsub = qmax
                qsurf = s1 - smax
            else:
                et = pet * (s1 / smax)
                qsub = qmax * exp(-f * (smax - s1))
                qsurf = 0
        """
        f = f / 10  # (0, 0.1)
        smax = smax * 1500  # (100, 1500)
        qmax = qmax * 50  # (10, 50)

        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)
        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)

        return [et, qsub, qsurf]

    def step_do(self, step_in, states):  # Define step function for the RNN
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket

        # Load the current input column
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]

        # Partition precipitation into rain and snow
        [_ps, _pr] = self.rainsnowpartition(p, t, self.tmin)
        # Snow bucket
        _m = self.snowbucket(s0, t, self.ddf, self.tmax)
        # Soil bucket
        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        # Water balance equations
        _ds0 = _ps - _m
        _ds1 = _pr + _m - _et - _qsub - _qsurf

        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]

    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        dayl = inputs[:, :, 2:3]

        # Calculate PET using Hamon’s formulation
        pet = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)

        # Concatenate pprcp, tmean, and pet into a new input
        new_inputs = K.concatenate((prcp, tmean, pet), axis=-1)

        # Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        # Recursively calculate state variables by using RNN
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        # Calculate final process variables
        [ps, pr] = self.rainsnowpartition(prcp, tmean, self.tmin)
        m = self.snowbucket(s0, tmean, self.ddf, self.tmax)
        [et, qsub, qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        if self.mode == "normal":
            return qsub + qsurf
        elif self.mode == "analysis":
            return K.concatenate([ps, pr, s0, s1, m, et, qsub + qsurf, qsub, qsurf], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 1)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 9)
