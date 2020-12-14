import numpy as np
from alibi_detect.cd import MMDDrift
from alibi_detect.utils.kernels import gaussian_kernel

def MMD_test(source_data,target_data,p_val=0.05,preprocess_kwargs={},chunk_size=100,
    n_permutations=20):
    """
    Functional wrapper around alibi_detect MMDDrift class, uses gaussian kernel
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
    Repeatedly carry out the MMD test, subsampling the data each time.  Returns mean and standard
    deviation of the p_values
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
    return np.mean(p_array),np.std(p_array)
    




