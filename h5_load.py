#coding: utf-8

#------------------------------------------------------------------------
# Description: load keras model weights HDF5 file
# Date: 2018.6
#------------------------------------------------------------------------

import h5py
import numpy as np

# check version
# python 3.6.4 (64bit) on win32
# windows 10 (64bit) 
# numpy (1.14.0)
# h5py (2.7.1)

def irekae2(w1):
    # coefficient position change (a,a)->(a',a')
    w2=np.zeros( w1.shape, dtype=np.float32)
    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
            w2[i,j]=w1[(w1.shape[0]-1)-i, (w1.shape[1]-1)-j]
    return w2

def irekae4(w1):  
    # coefficient position changeconv (*,*,a,a)->(*.*,a',a')
    w2=np.zeros( w1.shape, dtype=np.float32)
    for k in range(w1.shape[0]):
        for l in range(w1.shape[1]):
            for i in range(w1.shape[2]):
                for j in range(w1.shape[3]):
                    w2[k,l,i,j]=w1[k,l,(w1.shape[2]-1)-i, (w1.shape[3]-1)-j]
    return w2


class Class_net_from_h5_CNN(object):
    def __init__(self, IN_FILE='data/music_tagger_cnn_weights_theano.h5'):
        """
        Download music-auto_tagging-keras's CNN weights file of
        <https://github.com/keunwoochoi/music-auto_tagging-keras/tree/master/data/music_tagger_cnn_weights_theano.h5>
        and put it in the 'data/' directory.
        
        Contents:
        IN_FILE='data/music_tagger_cnn_weights_theano.h5' 
        ['bn1', 'bn2', 'bn3', 'bn4', 'bn5', 'bn_0_freq', 
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 
        'dropout1', 'dropout2', 'dropout3', 'dropout4', 'dropout5',
        'elu_1', 'elu_2', 'elu_3', 'elu_4', 'elu_5', 
        'flatten_1', 'input_1', 'output', 
        'pool1', 'pool2', 'pool3', 'pool4', 'pool5']
        bn1 ['bn1_beta', 'bn1_gamma', 'bn1_running_mean', 'bn1_running_std']
        conv1 ['conv1_W', 'conv1_b']
        output ['output_W', 'output_b']
        """
        self.IN_FILE = IN_FILE
        self.model_weights = h5py.File(self.IN_FILE, 'r')
        
        
    @property
    def norm0_b(self):
        return self.model_weights['bn_0_freq/bn_0_freq_beta'].value.reshape(self.model_weights['bn_0_freq/bn_0_freq_beta'].value.size)
    @property
    def norm0_g(self):
        return self.model_weights['bn_0_freq/bn_0_freq_gamma'].value.reshape(self.model_weights['bn_0_freq/bn_0_freq_gamma'].value.size)
    @property
    def norm0_m(self):
        return self.model_weights['bn_0_freq/bn_0_freq_running_mean'].value.reshape(self.model_weights['bn_0_freq/bn_0_freq_running_mean'].value.size)
    @property
    def norm0_v(self):
        return self.model_weights['bn_0_freq/bn_0_freq_running_std'].value.reshape(self.model_weights['bn_0_freq/bn_0_freq_running_std'].value.size)
    
    @property
    def conv1_W(self):
        return irekae4(self.model_weights['conv1/conv1_W'].value)
    @property
    def conv1_b(self):
        return self.model_weights['conv1/conv1_b'].value.reshape(self.model_weights['conv1/conv1_b'].value.size)
    @property
    def norm1_b(self):
        return self.model_weights['bn1/bn1_beta'].value.reshape(self.model_weights['bn1/bn1_beta'].value.size)
    @property
    def norm1_g(self):
        return self.model_weights['bn1/bn1_gamma'].value.reshape(self.model_weights['bn1/bn1_gamma'].value.size)
    @property
    def norm1_m(self):
        return self.model_weights['bn1/bn1_running_mean'].value.reshape(self.model_weights['bn1/bn1_running_mean'].value.size)
    @property
    def norm1_v(self):
        return self.model_weights['bn1/bn1_running_std'].value.reshape(self.model_weights['bn1/bn1_running_std'].value.size)
    
    @property
    def conv2_W(self):
        return irekae4(self.model_weights['conv2/conv2_W'].value)
    @property
    def conv2_b(self):
        return self.model_weights['conv2/conv2_b'].value.reshape(self.model_weights['conv2/conv2_b'].value.size)
    @property
    def norm2_b(self):
        return self.model_weights['bn2/bn2_beta'].value.reshape(self.model_weights['bn2/bn2_beta'].value.size)
    @property
    def norm2_g(self):
        return self.model_weights['bn2/bn2_gamma'].value.reshape(self.model_weights['bn2/bn2_gamma'].value.size)
    @property
    def norm2_m(self):
        return self.model_weights['bn2/bn2_running_mean'].value.reshape(self.model_weights['bn2/bn2_running_mean'].value.size)
    @property
    def norm2_v(self):
        return self.model_weights['bn2/bn2_running_std'].value.reshape(self.model_weights['bn2/bn2_running_std'].value.size)
    
    @property
    def conv3_W(self):
        return irekae4(self.model_weights['conv3/conv3_W'].value)
    @property
    def conv3_b(self):
        return self.model_weights['conv3/conv3_b'].value.reshape(self.model_weights['conv3/conv3_b'].value.size)
    @property
    def norm3_b(self):
        return self.model_weights['bn3/bn3_beta'].value.reshape(self.model_weights['bn3/bn3_beta'].value.size)
    @property
    def norm3_g(self):
        return self.model_weights['bn3/bn3_gamma'].value.reshape(self.model_weights['bn3/bn3_gamma'].value.size)
    @property
    def norm3_m(self):
        return self.model_weights['bn3/bn3_running_mean'].value.reshape(self.model_weights['bn3/bn3_running_mean'].value.size)
    @property
    def norm3_v(self):
        return self.model_weights['bn3/bn3_running_std'].value.reshape(self.model_weights['bn3/bn3_running_std'].value.size)
    
    @property
    def conv4_W(self):
        return irekae4(self.model_weights['conv4/conv4_W'].value)
    @property
    def conv4_b(self):
        return self.model_weights['conv4/conv4_b'].value.reshape(self.model_weights['conv4/conv4_b'].value.size)
    @property
    def norm4_b(self):
        return self.model_weights['bn4/bn4_beta'].value.reshape(self.model_weights['bn4/bn4_beta'].value.size)
    @property
    def norm4_g(self):
        return self.model_weights['bn4/bn4_gamma'].value.reshape(self.model_weights['bn4/bn4_gamma'].value.size)
    @property
    def norm4_m(self):
        return self.model_weights['bn4/bn4_running_mean'].value.reshape(self.model_weights['bn4/bn4_running_mean'].value.size)
    @property
    def norm4_v(self):
        return self.model_weights['bn4/bn4_running_std'].value.reshape(self.model_weights['bn4/bn4_running_std'].value.size)
    
    @property
    def conv5_W(self):
        return irekae4(self.model_weights['conv5/conv5_W'].value)
    @property
    def conv5_b(self):
        return self.model_weights['conv5/conv5_b'].value.reshape(self.model_weights['conv5/conv5_b'].value.size)
    @property
    def norm5_b(self):
        return self.model_weights['bn5/bn5_beta'].value.reshape(self.model_weights['bn5/bn5_beta'].value.size)
    @property
    def norm5_g(self):
        return self.model_weights['bn5/bn5_gamma'].value.reshape(self.model_weights['bn5/bn5_gamma'].value.size)
    @property
    def norm5_m(self):
        return self.model_weights['bn5/bn5_running_mean'].value.reshape(self.model_weights['bn5/bn5_running_mean'].value.size)
    @property
    def norm5_v(self):
        return self.model_weights['bn5/bn5_running_std'].value.reshape(self.model_weights['bn5/bn5_running_std'].value.size)
    
    @property
    def fc1_W(self):
        # output tag 50
        return self.model_weights['output/output_W'].value.T
    @property
    def fc1_b(self):
        # output tag 50
        return self.model_weights['output/output_b'].value.reshape(self.model_weights['output/output_b'].value.size)



class Class_net_from_h5_CRNN(object):
    def __init__(self, IN_FILE='data/crnn_net_gru_adam_ours_epoch_40.h5'):
        """"
        Download Music-Genre-Classification-with-Deep-Learning's CRNN weights file of
        <https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning/tree/master/models_trained/example_model/weights/crnn_net_gru_adam_ours_epoch_40.h5>
        And put it in the 'data/' directory.
        
        Or,
        Download music-auto_tagging-keras's CRNN weights file of
        <https://github.com/keunwoochoi/music-auto_tagging-keras/tree/master/data/music_tagger_crnn_weights_theano.h5>
        and put it in the 'data/' directory.
        
        Contents:
        
        IN_FILE='data/crnn_net_gru_adam_ours_epoch_40.h5'
        ['bn1', 'bn2', 'bn3', 'bn4', 'bn_0_freq', 
        'conv1', 'conv2', 'conv3', 'conv4', 
        'dropout1', 'dropout2', 'dropout3', 'dropout4', 
        'elu_1', 'elu_2', 'elu_3', 'elu_4', 'final_drop', 'gru1', 'gru2', 
        'input_1', 'permute_1', 'pool1', 'pool2', 'pool3', 'pool4', 'preds', 
        'reshape_1', 'zeropadding2d_1']
        
        
        IN_FILE='data/music_tagger_crnn_weights_theano.h5'
        ['bn1', 'bn2', 'bn3', 'bn4', 'bn_0_freq', 
        'conv1', 'conv2', 'conv3', 'conv4', 
        'dropout1', 'dropout2', 'dropout3', 'dropout4', 'dropout_1', 
        'elu_6', 'elu_7', 'elu_8', 'elu_9', 
        'gru1', 'gru2', 'input_2', 'output', 'permute_1', 
        'pool1', 'pool2', 'pool3', 'pool4', 'reshape_1', 'zeropadding2d_1']
        ['bn1_beta', 'bn1_gamma', 'bn1_running_mean', 'bn1_running_std']
        ['bn_0_freq_beta', 'bn_0_freq_gamma', 'bn_0_freq_running_mean', 'bn_0_freq_running_std']
        ['conv1_W', 'conv1_b']
        gru1  ['gru1_U_h', 'gru1_U_r', 'gru1_U_z', 
               'gru1_W_h', 'gru1_W_r', 'gru1_W_z', 
               'gru1_b_h', 'gru1_b_r', 'gru1_b_z']
        output ['output_W', 'output_b']
        
        """
        self.IN_FILE = IN_FILE
        self.model_weights = h5py.File(self.IN_FILE, 'r')
        
    
    @property
    def norm0_b(self):
        return self.model_weights['bn_0_freq/bn_0_freq_beta'].value.reshape(self.model_weights['bn_0_freq/bn_0_freq_beta'].value.size)
    @property
    def norm0_g(self):
        return self.model_weights['bn_0_freq/bn_0_freq_gamma'].value.reshape(self.model_weights['bn_0_freq/bn_0_freq_gamma'].value.size)
    @property
    def norm0_m(self):
        return self.model_weights['bn_0_freq/bn_0_freq_running_mean'].value.reshape(self.model_weights['bn_0_freq/bn_0_freq_running_mean'].value.size)
    @property
    def norm0_v(self):
        return self.model_weights['bn_0_freq/bn_0_freq_running_std'].value.reshape(self.model_weights['bn_0_freq/bn_0_freq_running_std'].value.size)
    
    @property
    def conv1_W(self):
        return irekae4( self.model_weights['conv1/conv1_W'].value)
    @property
    def conv1_b(self):
        return self.model_weights['conv1/conv1_b'].value.reshape(self.model_weights['conv1/conv1_b'].value.size)
    @property
    def norm1_b(self):
        return self.model_weights['bn1/bn1_beta'].value.reshape(self.model_weights['bn1/bn1_beta'].value.size)
    @property
    def norm1_g(self):
        return self.model_weights['bn1/bn1_gamma'].value.reshape(self.model_weights['bn1/bn1_gamma'].value.size)
    @property
    def norm1_m(self):
        return self.model_weights['bn1/bn1_running_mean'].value.reshape(self.model_weights['bn1/bn1_running_mean'].value.size)
    @property
    def norm1_v(self):
        return self.model_weights['bn1/bn1_running_std'].value.reshape(self.model_weights['bn1/bn1_running_std'].value.size)
    
    @property
    def conv2_W(self):
        return irekae4(self.model_weights['conv2/conv2_W'].value)
    @property
    def conv2_b(self):
        return self.model_weights['conv2/conv2_b'].value.reshape(self.model_weights['conv2/conv2_b'].value.size)
    @property
    def norm2_b(self):
        return self.model_weights['bn2/bn2_beta'].value.reshape(self.model_weights['bn2/bn2_beta'].value.size)
    @property
    def norm2_g(self):
        return self.model_weights['bn2/bn2_gamma'].value.reshape(self.model_weights['bn2/bn2_gamma'].value.size)
    @property
    def norm2_m(self):
        return self.model_weights['bn2/bn2_running_mean'].value.reshape(self.model_weights['bn2/bn2_running_mean'].value.size)
    @property
    def norm2_v(self):
        return self.model_weights['bn2/bn2_running_std'].value.reshape(self.model_weights['bn2/bn2_running_std'].value.size)
    
    @property
    def conv3_W(self):
        return irekae4(self.model_weights['conv3/conv3_W'].value)
    @property
    def conv3_b(self):
        return self.model_weights['conv3/conv3_b'].value.reshape(self.model_weights['conv3/conv3_b'].value.size)
    @property
    def norm3_b(self):
        return self.model_weights['bn3/bn3_beta'].value.reshape(self.model_weights['bn3/bn3_beta'].value.size)
    @property
    def norm3_g(self):
        return self.model_weights['bn3/bn3_gamma'].value.reshape(self.model_weights['bn3/bn3_gamma'].value.size)
    @property
    def norm3_m(self):
        return self.model_weights['bn3/bn3_running_mean'].value.reshape(self.model_weights['bn3/bn3_running_mean'].value.size)
    @property
    def norm3_v(self):
        return self.model_weights['bn3/bn3_running_std'].value.reshape(self.model_weights['bn3/bn3_running_std'].value.size)
    
    @property
    def conv4_W(self):
        return irekae4(self.model_weights['conv4/conv4_W'].value)
    @property
    def conv4_b(self):
        return self.model_weights['conv4/conv4_b'].value.reshape(self.model_weights['conv4/conv4_b'].value.size)
    @property
    def norm4_b(self):
        return self.model_weights['bn4/bn4_beta'].value.reshape(self.model_weights['bn4/bn4_beta'].value.size)
    @property
    def norm4_g(self):
        return self.model_weights['bn4/bn4_gamma'].value.reshape(self.model_weights['bn4/bn4_gamma'].value.size)
    @property
    def norm4_m(self):
        return self.model_weights['bn4/bn4_running_mean'].value.reshape(self.model_weights['bn4/bn4_running_mean'].value.size)
    @property
    def norm4_v(self):
        return self.model_weights['bn4/bn4_running_std'].value.reshape(self.model_weights['bn4/bn4_running_std'].value.size)
    
    @property
    def gru1_W_r(self):
        return self.model_weights['gru1/gru1_W_r'].value.T
    @property
    def gru1_W_z(self):
        return self.model_weights['gru1/gru1_W_z'].value.T
    @property
    def gru1_W(self):
        return self.model_weights['gru1/gru1_W_h'].value.T
    @property
    def gru1_b_r(self):
        return self.model_weights['gru1/gru1_b_r'].value.reshape(self.model_weights['gru1/gru1_b_r'].value.size)
    @property
    def gru1_b_z(self):
        return self.model_weights['gru1/gru1_b_z'].value.reshape(self.model_weights['gru1/gru1_b_z'].value.size)
    @property
    def gru1_b(self):
        return self.model_weights['gru1/gru1_b_h'].value.reshape(self.model_weights['gru1/gru1_b_h'].value.size)
    @property
    def gru1_U_r(self):
        return self.model_weights['gru1/gru1_U_r'].value.T
    @property
    def gru1_U_z(self):
        return self.model_weights['gru1/gru1_U_z'].value.T
    @property
    def gru1_U(self):
        return self.model_weights['gru1/gru1_U_h'].value.T
    
    @property
    def gru2_W_r(self):
        return self.model_weights['gru2/gru2_W_r'].value.T
    @property
    def gru2_W_z(self):
        return self.model_weights['gru2/gru2_W_z'].value.T
    @property
    def gru2_W(self):
        return self.model_weights['gru2/gru2_W_h'].value.T
    @property
    def gru2_b_r(self):
        return self.model_weights['gru2/gru2_b_r'].value.reshape(self.model_weights['gru2/gru2_b_r'].value.size)
    @property
    def gru2_b_z(self):
        return self.model_weights['gru2/gru2_b_z'].value.reshape(self.model_weights['gru2/gru2_b_z'].value.size)
    @property
    def gru2_b(self):
        return self.model_weights['gru2/gru2_b_h'].value.reshape(self.model_weights['gru2/gru2_b_h'].value.size)
    @property
    def gru2_U_r(self):
        return self.model_weights['gru2/gru2_U_r'].value.T
    @property
    def gru2_U_z(self):
        return self.model_weights['gru2/gru2_U_z'].value.T
    @property
    def gru2_U(self):
        return self.model_weights['gru2/gru2_U_h'].value.T
    
    @property
    def fc1_W(self):
        if self.IN_FILE == "data/music_tagger_crnn_weights_theano.h5":
            # output tag 50
            return self.model_weights['output/output_W'].value.T
        else:
            # output tag 10
            return self.model_weights['preds/preds_W'].value.T
    @property
    def fc1_b(self):
        if self.IN_FILE == "data/music_tagger_crnn_weights_theano.h5":
            # output tag 50
            return self.model_weights['output/output_b'].value.reshape(self.model_weights['output/output_b'].value.size)
        else:
            # output tag 10
            return self.model_weights['preds/preds_b'].value.reshape(self.model_weights['preds/preds_b'].value.size)


if __name__ == '__main__':
    
    IN_FILE='data/crnn_net_gru_adam_ours_epoch_40.h5'
    model_weights = h5py.File(IN_FILE, 'r')
    #print (model_weights.keys())
    print (list(model_weights))  # abc order
    
    print (list(model_weights['bn_0_freq'].keys()))
    print (list(model_weights['conv1'].keys()))
    print (list(model_weights['gru1'].keys()))
    print (list(model_weights['bn1'].keys()))
    
    print ('CRNN; bn_1: mean, var, gamma, beta')
    a=model_weights['bn1/bn1_running_mean'].value
    b=model_weights['bn1/bn1_running_std'].value
    c=model_weights['bn1/bn1_gamma'].value
    d=model_weights['bn1/bn1_beta'].value
    print ('shape',a.shape)
    print (a)
    print (b)
    print (c)
    print (d)
