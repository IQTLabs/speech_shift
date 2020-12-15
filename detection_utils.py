import numpy as np
from alibi_detect.cd import MMDDrift
from alibi_detect.utils.kernels import gaussian_kernel

def MMD_test(source_data,target_data,p_val=0.05,preprocess_kwargs={},chunk_size=100,
    n_permutations=20):
    """
    Functional wrapper around alibi_detect MMDDrift class that uses uses gaussian kernel
    (https://docs.seldon.io/projects/alibi-detect/en/stable/api/alibi_detect.cd.mmd.html)
    

    Inputs:
        source_data - numpy.ndarray of shape (number of source samples,embedding dimension),
            samples from the source distribution
        target_data - numpy.ndarray of shape (number of target samples,embedding dimension),
            samples from the target distribution
        p_val - p-value used for the significance of the permutation test.
        preprocess_kwargs - Kwargs for a preprocessing function, pass callables under "model" key
        chunk_size - Chunk size if dask is used to parallelise the computation.
        n_permutations - Number of permutations used in the permutation test.
    Outputs:
        p - float, empirical p-value determined using the permutation test
    """
    source_size,source_dim = np.shape(source_data)
    target_size,target_dim = np.shape(target_data)
    assert source_dim==target_dim, "Source dimension must match target dimension"
    cd = MMDDrift(
        p_val=p_val,
        X_ref = source_data,
        preprocess_kwargs=preprocess_kwargs,
        kernel=gaussian_kernel,
        chunk_size=chunk_size,
        n_permutations=n_permutations
    )
    result = cd.predict(target_data,return_p_val=True)
    return result['data']['p_val']

def repeated_MMD_test(source_data,target_data,p_val=0.05,preprocess_kwargs={},chunk_size=100,
    n_permutations=20,n_samples=100,n_splits=5):
    """
    Repeatedly carry out the MMD test, subsampling the data each time.  Returns an array of p-values
    Inputs:
        source_data - numpy.ndarray of shape (number of source samples,embedding dimension),
            samples from the source distribution
        target_data - numpy.ndarray of shape (number of target samples,embedding dimension),
            samples from the target distribution
        p_val - p-value used for the significance of the permutation test.
        preprocess_kwargs - Kwargs for a preprocessing function, pass callables under "model" key
        chunk_size - Chunk size if dask is used to parallelise the computation.
        n_permutations - Number of permutations used in the permutation test.
        n_samples - number of samples to use from the source and target data in each subsampling
        n_splits - number of different subsamplings to carry out
    Outputs:
        p_array - np.ndarray of shape (n_splits,), the set of p-values computed
    """
    source_size,source_dim = np.shape(source_data)
    target_size,target_dim = np.shape(target_data)
    n_samples = min([n_samples,source_size,target_size])
    p_list = []
    for i in range(n_splits):
        source_rows = np.random.choice(source_size,size=n_samples,replace=False)
        target_rows = np.random.choice(target_size,size=n_samples,replace=False)
        p_temp = MMD_test(source_data[source_rows,:],target_data[target_rows,:],p_val=p_val,chunk_size=chunk_size,
        preprocess_kwargs=preprocess_kwargs,n_permutations=n_permutations)
        p_list.append(p_temp)
    p_array = np.array(p_list)
    return p_array
    




