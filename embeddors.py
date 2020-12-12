import numpy as np
import wget

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_hub as hub
import os
import torch
from fairseq.models.wav2vec import Wav2VecModel



class Wav2VecEmbeddor:
    def __init__(self,weight_path=None,use_cpu=True):
        """
        Initialize an embeddor that uses the Wav2Vec model.

        Inputs:
            weight_path - path to an instance of pt file corresponding to the 
            wav2vec_large model
            use_cpu - boolean, whether to use cpu or gpu
        """
        if weight_path is None:
            print('Downloading wav2vec model')
            if not os.path.exists('models'):
                os.makedirs('models')
            url = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt'
            wget.download(url,'models/wav2vec_large.pt')
            weight_path = 'models/wav2vec_large.pt'
        if use_cpu:
            cp = torch.load(weight_path,map_location=torch.device('cpu'))
        else:
            cp = torch.load(weight_path)
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()
        
    def embed(self,waveform):
        """
        Convert a scalar waveform into a sequence of latent vectors using the 
        Wav2Vec model.

        Inputs:
            waveform - A numpy ndarray of shape (1,number of samples) or (number of samples,)
                represents the audio waveform sampled at 16 Khz
        Outputs:
            embedding_sequence - a numpy ndarray of shape (1,512,duration)
        """
        
        if np.ndim(waveform) < 2:
            wave_temp = np.expand_dims(waveform,0)
            waveform_pt = torch.from_numpy(wave_temp)
        else:
            waveform_pt = torch.from_numpy(waveform)
        z = self.model.feature_extractor(waveform_pt.float())
        c = self.model.feature_aggregator(z)
        embedding_sequence = c.detach().cpu().numpy()

        return embedding_sequence


class TrillEmbeddor:
    def __init__(self):
        self.module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/2')
    def embed(self,waveform):
        """
        Convert a scalar waveform into a sequence of latent vectors using the 
        Trill model.

        Inputs:
            waveform - A numpy ndarray of shape (1,number of samples) or (number of samples,)
                represents the audio waveform sampled at 16 Khz
        Outputs:
            embedding_sequence - a numpy ndarray of shape (1,2048,duration)
        """
        if np.ndim(waveform) < 2:
            samples_t = np.expand_dims(waveform,0)
        else:
            samples_t = np.copy(waveform)
        emb = self.module(samples=samples_t, sample_rate=16000)['embedding']
        # Flip last two dimensions of emb
        rearranged = tf.transpose(emb,perm=[0,2,1])
        return rearranged.numpy()


