import numpy as np

def md(detections, norm_distance, top_n):
    """
    Calculate the vector cosine similarity score and return the sum of the first top_n scores of each vector.
    
    Args:
        detections: Array data
        norm_distance: Standardized distance value (scalar) in the new dimension
        top_n: The number of highest similarity scores retained
    
    Returns:
        List of [score_sum, original_vector, index]
    """
    # Input verification: check whether detections is empty or whether top_n is valid
    if detections is None or detections.size == 0 or top_n <= 0:
        return []
    
    # Make sure the input is a NumPy array
    dets_array = np.array(detections, dtype=float)
    n_samples, n_features = dets_array.shape
    
    # Precomputed constant term
    nd_squared = norm_distance ** 2  
    denominator_left = norm_distance  
    
    # Result storage
    result_list = []
    
    # Vectorize each vector
    for i in range(n_samples):
        current_vector = dets_array[i]
        
        # Calculate the sum of squares of the difference between all vectors and the current vector
        diff = dets_array - current_vector  
        diff_squared_sum = np.sum(diff ** 2, axis=1)  
        
        # Calculate the right part of the denominator
        denominator_right = np.sqrt(diff_squared_sum + nd_squared)  
        denominator = denominator_left * denominator_right  
        
        # Calculate the molecule
        numerator = nd_squared * np.ones(n_samples)  
        
        # Calculate the similarity score and avoid dividing by zero
        similarity_scores = np.where(denominator == 0, 0, numerator / denominator)
        similarity_scores[i] = -np.inf  
        
        # Get the sum of the top_n scores
        top_scores_sum = np.sum(np.partition(similarity_scores, -top_n)[-top_n:])
        
        # Save result
        result_list.append([top_scores_sum, detections[i].tolist(), i])
    
    return result_list
