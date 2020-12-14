import embeddors
import aggregators
import numpy as np
import librosa
import tqdm

def pipeline(wav_list=None,file_list=None,embeddor=None,aggregator=None):
    """
    A utility function for converting a list of waveforms or .wav files
    into fixed length vector embeddings.

    Inputs:
        wav_list - list of waveforms, numpy arrays of shape (1,num samples)
        file_list - list of .wav files to use.  If wav_list is none, this
            argument must be supplied
        embeddor - the embedding module, must implement an embed method that
            converts waveform array to a an embedding sequence of shape (1,embedding dim,duration)
        aggregrator - the aggregator module, must implement an aggregation method
            that converts an embedding sequence to a numpy array of shape (1,embedding dim)
    Outputs:
        embedding_matrix - a numpy array of shape (number of waveforms, embedding dim)
        containing all the fixed length embeddings. 
    """
    if embeddor is None:
        # default is Wav2Vec embedding
        embeddor = embeddors.Wav2VecEmbeddor()
    if aggregator is None:
        # default is mean aggregation
        aggregator = aggregators.MeanAggregator()
    if wav_list is None:
        print("Converting files to waveforms")
        wav_list = [librosa.load(file_name,sr=16000)[0] for file_name in file_list]
    def converter(waveform):
        emb_seq = embeddor.embed(waveform)
        return aggregator.aggregate(emb_seq)
    vec_list = list(map(converter,tqdm.tqdm(wav_list)))
    return np.concatenate(vec_list)

