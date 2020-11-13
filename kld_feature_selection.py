# discrete form
def kld_feature_selection(feature_set: np.array, target_set: np.array, threshold = 1):
    """
    Purpose:
        Computes discrete KL divergence for feature selection. Only supports binary classes for now.
        
    Args:
        feature_set: Data containing all independent variables
        target_set: Data containing output variable (0 or 1)
        threshold: Threshold based on standard-scaled scores that KLD must pass in order to keep a feature. Higher the value, less features will be kept
        
    Returns:
        Index of features that have significant KL divergence (based off standardized scores)
        
    Dependencies:
        scipy, numpy
    """
    
    # if target_set is not a column vector already, reshape it
    target_set = target_set.reshape(-1,1) # calling this multiple times won't affect the vector
    
    # list of KL divergence to return
    divergence_list = []
    
    
    full_data = np.concatenate((target_set, feature_set), axis=1)
    label_0_full = full_data[full_data[:,0] == 0]
    label_1_full = full_data[full_data[:,0] == 1]
    
    # make sample sizes equal (minimum of two groups)
    sample_size = min(label_0_full.shape[0], label_1_full.shape[0])
    
    for i in range(feature_set.shape[1]):
        if (i == feature_set.shape[1] - 1):
            continue
        else:
            label_0 = np.random.choice(label_0_full[:,i+1], size=sample_size)
            label_1 = np.random.choice(label_1_full[:,i+1], size=sample_size)
            
            # establish lower/upper bounds
            lower_bound = min(min(label_0), min(label_1))
            upper_bound = max(max(label_0), max(label_1))

            # create 100 data points to evaluate densities
            data_points = np.linspace(lower_bound, upper_bound, 100)

            # densities from two groups
            p_kde = gaussian_kde(label_0).evaluate(data_points)
            q_kde = gaussian_kde(label_1).evaluate(data_points)

            # KLD is not symmetric, so take the minimul of KL(P||Q) and KL(Q||P)

            kld = min(
                    np.sum(np.where(p_kde != 0, p_kde * np.log(p_kde / q_kde), 0)),
                    np.sum(np.where(q_kde != 0, q_kde * np.log(q_kde / p_kde), 0))
                )

            divergence_list.append(kld)
            
    divergence_array = np.array(divergence_list)
    standardized_divergence = abs((divergence_array - np.mean(divergence_array)) / np.std(divergence_array))
    standardized_divergence = standardized_divergence.reshape(-1,1)
    index_array = np.arange(0,feature_set.shape[1]-1).reshape(-1,1)
    
    divergence_matrix = np.concatenate((index_array, standardized_divergence), axis=1)
    significant_divergence = divergence_matrix[divergence_matrix[:,1] > threshold]
    return significant_divergence[:,0].astype('int')
