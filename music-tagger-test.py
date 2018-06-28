#coding: utf-8

#---------------------------------------------------------------------------------------------
#   Description: a remake of Music Genre Classification with Deep Learning for Chainer
#   Date: 2018.6
#---------------------------------------------------------------------------------------------
#   This is based on
#         audio_processor.py, example_tagging.py music_tagger_cnn.py and music_tagger_crnn.py
#         of music-auto_tagging-keras <https://github.com/keunwoochoi/music-auto_tagging-keras>
#         Copyright (c) 2016 Keunwoo Choi.
#         Pls see LICENSE-music-auto_tagging-keras.md in the 'docs' directory
#     and 
#         quick_test.py, tagger_net.py, train_tagger_net.py and utils.py
#         of Music Genre Classification with Deep Learning <https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning>
#----------------------------------------------------------------------------------------------

import argparse
import numpy as np

import chainer
from chainer import Chain, serializers, optimizers, cuda
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import librosa
from librosa import feature, core, filters
from gru2 import *
from batch_normalization2 import *


# check version
# python 3.6.4 (64bit) on win32
# windows 10 (64bit) 
# chainer (3.2.0)
# numpy (1.14.0)
# librosa (0.6.0)

# CRNN
class MusicTaggerCRNN(Chain):
    def __init__(self, net=None):
        super(MusicTaggerCRNN, self).__init__()
        with self.init_scope():
            # input_shape = (1, 96, 1440)  (1, mel-bands, time frames)
            self.norm0 = BatchNormalization2(1440, initial_gamma=net.norm0_g if net else None ,initial_beta=net.norm0_b if net else None,
                                                   initial_avg_mean=net.norm0_m if net else None, initial_avg_var=net.norm0_v if net else None)
            # conv1 1->64
            self.conv1 = L.Convolution2D(1, 64, 3, pad=1, initialW=net.conv1_W if net else None ,initial_bias=net.conv1_b if net else None)
            self.norm1 = BatchNormalization2(64, initial_gamma=net.norm1_g if net else None ,initial_beta=net.norm1_b if net else None,
                                                  initial_avg_mean=net.norm1_m if net else None, initial_avg_var=net.norm1_v if net else None)
            # conv2 64->128
            self.conv2 = L.Convolution2D(64, 128, 3, pad=1, initialW=net.conv2_W if net else None ,initial_bias=net.conv2_b if net else None)
            self.norm2 = BatchNormalization2(128, initial_gamma=net.norm2_g if net else None ,initial_beta=net.norm2_b if net else None,
                                                   initial_avg_mean=net.norm2_m if net else None, initial_avg_var=net.norm2_v if net else None)
            # conv3 128->128
            self.conv3 = L.Convolution2D(128, 128, 3, pad=1, initialW=net.conv3_W if net else None ,initial_bias=net.conv3_b if net else None)
            self.norm3 = BatchNormalization2(128, initial_gamma=net.norm3_g if net else None ,initial_beta=net.norm3_b if net else None,
                                                   initial_avg_mean=net.norm3_m if net else None, initial_avg_var=net.norm3_v if net else None)
            # conv4 128->128
            self.conv4 = L.Convolution2D(128, 128, 3, pad=1, initialW=net.conv4_W if net else None ,initial_bias=net.conv4_b if net else None)
            self.norm4 = BatchNormalization2(128, initial_gamma=net.norm4_g if net else None ,initial_beta=net.norm4_b if net else None,
                                                   initial_avg_mean=net.norm4_m if net else None, initial_avg_var=net.norm4_v if net else None)
            
            # GRU1 (statefull)  128->32
            self.gru1  = StatefulGRU2(128, 32, init=net.gru1_W if net else None,     inner_init=net.gru1_U if net else None,     bias_init=net.gru1_b if net else None,
                                               init_r=net.gru1_W_r if net else None, inner_init_r=net.gru1_U_r if net else None, bias_init_r=net.gru1_b_r if net else None,
                                               init_z=net.gru1_W_z if net else None, inner_init_z=net.gru1_U_z if net else None, bias_init_z=net.gru1_b_z if net else None )
            # GRU2 (statefull)   32->32
            self.gru2  = StatefulGRU2(32, 32,  init=net.gru2_W if net else None,     inner_init=net.gru2_U if net else None,     bias_init=net.gru2_b if net else None,
                                               init_r=net.gru2_W_r if net else None, inner_init_r=net.gru2_U_r if net else None, bias_init_r=net.gru2_b_r if net else None,
                                               init_z=net.gru2_W_z if net else None, inner_init_z=net.gru2_U_z if net else None, bias_init_z=net.gru2_b_z if net else None )
            # full connection 32->10 (32->50)
            self.fc1 = L.Linear(32, 10, initialW=net.fc1_W if net else None ,initial_bias=net.fc1_b if net else None)
        
    def __call__(self, X):
        h0 = F.pad(X, ((0,0),(0,0),(0,0),(37,37)), 'constant')  # (1, 96, 1366) -> (1, 96, 1440)
        h1 = F.transpose( self.norm0( F.transpose(h0,axes=(0, 3 , 1, 2)) ), axes=(0, 2 , 3, 1) )  # normalize along time axis is OK?
        h1 = F.max_pooling_2d( F.elu(self.norm1(self.conv1(h1))), (2,2), stride=(2,2) )
        h1 = F.dropout(h1, ratio=0.1)
        h2 = F.max_pooling_2d( F.elu(self.norm2(self.conv2(h1))), (3,3), stride=(3,3) )
        h2 = F.dropout(h2, ratio=0.1) 
        h3 = F.max_pooling_2d( F.elu(self.norm3(self.conv3(h2))), (4,4), stride=(4,4) )
        h3 = F.dropout(h3, ratio=0.1) 
        h4 = F.max_pooling_2d( F.elu(self.norm4(self.conv4(h3))), (4,4), stride=(4,4) )
        h4 = F.dropout(h4, ratio=0.1)
        h4 =  F.transpose( h4, axes=(0, 3 , 1, 2) )
        h4 =  F.reshape( h4, (h4.shape[0], 15,128) )
        
        self.gru1.reset_state() # reset hidden states per. track Is this OK?
        self.gru2.reset_state() # reset hidden states per. track Is this OK?
        for i in range (h4.shape[1]): 
            h5 = self.gru1(h4[:,i,:])
            h6 = self.gru2(h5)
        
        h6 = F.dropout(h6, ratio=0.3) 
        h7 = F.sigmoid(self.fc1(h6))
        
        return h7
    
    def load(self, fname="data/crnn.model"):
        serializers.load_npz(fname, self)
    
    def save(self, fname="data/crnn.model"):
        serializers.save_npz(fname, self)


def compute_melgram(audio_path, SR=12000, N_FFT=512, N_MELS=96, HOP_LEN=256, DURA=29.12): # compute only center portion of the track
    """
    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..
    """
    print ('loading...', audio_path)
    src, sr = librosa.load(audio_path, sr=SR)  # load whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))  # still problem ?
    elif n_sample > n_sample_fit:  # if too long
        sp0=int((n_sample-n_sample_fit)/2)
        src = src[sp0 : sp0 + n_sample_fit ]
        
        
    # feature.melspectrogram out still power. Is use amplitude_to_db OK?  Or, is it power_to_db?
    melgram= feature.melspectrogram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS)
    ret= core.amplitude_to_db(melgram, ref=1.0)
    
    """
    # alternative:
    power=2
    S = np.abs( core.stft(y=src, n_fft=N_FFT, hop_length=HOP_LEN)  ) **power
    mel_basis = filters.mel(sr, n_fft=N_FFT, n_mels=N_MELS)
    ret= np.dot(mel_basis, S)
    ret= core.power_to_db(ret, ref=1.0) # mel_basis is still power
    ret= core.amplitude_to_db(ret, ref=1.0) # mel_basis is still power
    """
    ret = ret[np.newaxis, np.newaxis, :]
    return ret

def load_wav_and_get_melgrams(audio_paths):
    melgrams = np.zeros((0, 1, 96, 1366))
    for audio_path in audio_paths:
        melgram = compute_melgram(audio_path)
        melgrams = np.concatenate((melgrams, melgram), axis=0)
    return melgrams

def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]


def print_result(tags, pred_tags):
    print('* top-10 tags: genre prediction result percentage *')
    for song_idx, audio_path in enumerate(audio_paths):
        total=pred_tags[song_idx, :].sum()
        pred_tags[song_idx, :]=pred_tags[song_idx, :] / total * 100.
        sorted_result = sort_result(tags, pred_tags[song_idx, :].tolist())
        print(audio_path)
        print(sorted_result[:5])
        print(sorted_result[5:10])
        print(' ')


# music-auto_tagging-keras Tags 50
tag50 = ['rock', 'pop', 'alternative', 'indie', 'electronic',
        'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
        'beautiful', 'metal', 'chillout', 'male vocalists',
        'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
        '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
        'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
        'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
        '70s', 'party', 'country', 'easy listening',
        'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
        'Progressive rock', '60s', 'rnb', 'indie pop',
        'sad', 'House', 'happy']

# GTZAN Dataset: Tags10
tag10 = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Music Tagger for Chainer')
    parser.add_argument('--modelSel', '-m', default='CRNN', help='model select CRNN Or CNN')
    parser.add_argument('--en', action='store_false', help='add --en if use keras h5 weight data')
    parser.add_argument('--resume', '-r', default='data/crnn.model',help='Resume from model file ')
    args = parser.parse_args()

    # Set audio_path to genre prediction.
    audio_paths = ['data/bensound-thejazzpiano.wav',  # jazz, duration around 32sec, 44100Hz Mono
                   'data/bensound-actionable.wav']    # rock, duration around 32sec, 44100Hz Mono
    
    # load wav file and compute melgram
    melgrams2=np.array(load_wav_and_get_melgrams(audio_paths),dtype=np.float32)
    
    if args.resume and args.en:
        if args.modelSel == 'CRNN':
            model_cnn = MusicTaggerCRNN()
        else:
            model_cnn = MusicTaggerCNN()
        # Resume from model file
        print ('loading model file', args.resume)
        model_cnn.load(fname= args.resume)
    else:
        if args.modelSel == 'CRNN':
            from h5_load import Class_net_from_h5_CRNN
            net0=  Class_net_from_h5_CRNN()
            model_cnn = MusicTaggerCRNN(net=net0)
            # save the weights as model file
            model_cnn.save(fname="data/crnn.model")  
        else:
            from h5_load import Class_net_from_h5_CNN
            net0=  Class_net_from_h5_CNN()
            model_cnn = MusicTaggerCNN(net=net0)
            # save the weights as model file
            model_cnn.save(fname="data/cnn.model")
    
    # enter chainer into test mode (no train mode)
    with chainer.using_config('train', False):
         pred_tags=model_cnn(melgrams2)
    
    # show genre prediction result
    if args.modelSel == 'CRNN':
        print_result(tag10,pred_tags.data)
    else:
        print_result(tag50,pred_tags.data)

