import numpy as np

def extend_feature_vector(featureVec, alpha, rf):
    
    assert alpha >= 0
    #contanins values to to appended at the end of vector
    new_values = [] 
    #compute sq. sum
    sq_sum = 0
    for i in range(featureVec.shape[1]):
        sq_sum += featureVec[0][i]*featureVec[0][i]

    new_values.append(alpha)
    new_values.append(-alpha*sq_sum)
    new_values.append(alpha)
    new_values.append(0)
    new_values.append(rf)

    new_feature_vec = alpha*np.array(featureVec)
    # extended vector after appendin new values
    # extended_feature_vec = list(new_feature_vec) + new_values
    extended_feature_vec = []
    for r in range(featureVec.shape[1]):
        extended_feature_vec.append(new_feature_vec[0][r])
    extended_feature_vec += new_values
    return extended_feature_vec


import numpy as np


def extend_feature_vector_Q(featureVec, beta, rq, theta):
    assert beta >= 0
    # contanins values to to appended at the end of vector
    new_values = []
    # compute sq. sum
    sq_sum = 0
    for i in range(featureVec.shape[1]):
        sq_sum += featureVec[0][i]*featureVec[0][i]

    new_values.append(-beta * sq_sum)
    new_values.append(beta)
    # beta*theta^2
    new_values.append(beta * theta * theta)
    new_values.append(rq)
    new_values.append(0)

    new_feature_vec = 2 * beta * np.array(featureVec)
    # extended vector after appendin new values
    extended_feature_vec = []
    for r in range(featureVec.shape[1]):
        extended_feature_vec.append(new_feature_vec[0][r])
    extended_feature_vec += new_values

    return extended_feature_vec



def convert_sim_to_theta(sim, c):
    theta = (c - sim) / sim
    # round off till 2 decimals
    theta = round(theta, 2)
    return theta


def extend_feature_vector_DVREI(featureVec, lamda, beta):
    new_values = []
    sq_sum = 0
    new_values.append(lamda)

    new_feature_vec = lamda * np.array(featureVec)
    extended_feature_vec = []
    for r in range(featureVec.shape[1]):
        extended_feature_vec.append(new_feature_vec[0][r])
    extended_feature_vec += new_values
    for r in range(len(beta)):
        extended_feature_vec.append(beta[r])

    return extended_feature_vec
def extend_feature_vector_DVREI_T(featureVec, lamda, yibux):
    extended_feature_vec = []
    new_feature_vec = lamda * np.array(featureVec)
    for r in range(len(new_feature_vec)):
        extended_feature_vec.append(new_feature_vec[r] + yibux[r])

    return  extended_feature_vec