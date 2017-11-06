import numpy as np
from preprocessing_hmm_baum_welch_em import preprocessing,convert_given_observation_sequence_to_matrix_form
from print_functions import print_trans_emission_matrices, print_gamma_delta_probabilities, print_forward_backward_probabilities
from initialization_of_matrices import initialize_initial_probabilities,initialize_transition_matrix,initialize_emission_matrix

file_input = open("observations.txt",'r')

def forward_algorithm(initial_probabilities,transition_matrix,matrix_form_of_observations_inside_list):
    
    list_of_alphas_at_each_timestep=[]
    list_of_alphas_at_each_timestep.append(initial_probabilities)
    alpha_t_minus_1 = initial_probabilities
    transition_matrix_transpose = np.transpose(transition_matrix)
    
    #print "forward probabilities at each timestep"
    
    for observation_item in matrix_form_of_observations_inside_list:
        alpha_t = np.dot(np.dot(observation_item,transition_matrix_transpose),alpha_t_minus_1)
        list_of_alphas_at_each_timestep.append(alpha_t)
        #print alpha_t
        alpha_t_minus_1 = alpha_t
    
    return list_of_alphas_at_each_timestep

def backward_algorithm(endstate_probability,transition_matrix,matrix_form_of_observations_inside_list):
    
    list_of_betas_at_each_timestep=[]
    list_of_betas_at_each_timestep.append(endstate_probability)
    rev_observations_list = reversed(matrix_form_of_observations_inside_list)
    beta_t_plus_1 = endstate_probability
    #transition_matrix_transpose = np.transpose(transition_matrix) #have doubt.. check it
    
    #print "backward probabilities at each timestep"
    
    for observation_item in rev_observations_list:
        beta_t = np.dot(np.dot(transition_matrix,observation_item),beta_t_plus_1)
        list_of_betas_at_each_timestep.insert(0,beta_t)
        #print beta_t
        beta_t_plus_1 = beta_t 
    
    return list_of_betas_at_each_timestep

def gamma_values_compute(forward_prob_list,backward_prob_list, probability_of_sequence_given_theta):
    length = len(forward_prob_list)
    gamma_values_list = []
    for t in range(length):
        gamma_values_at_t = forward_prob_list[t]*backward_prob_list[t]
        normalized_gamma_values_at_t = gamma_values_at_t/probability_of_sequence_given_theta
        #print normalized_gamma_values_at_t
        gamma_values_list.append(normalized_gamma_values_at_t)
    return gamma_values_list

def delta_values_compute(forward_prob_list,backward_prob_list,transition_matrix,observations_inside_list, probability_of_sequence_given_theta):
    forward_prob_list = forward_prob_list[1:] #starting one is the initial prob
    backward_prob_list = backward_prob_list[1:] #starting one is the initial prob
    length = len(forward_prob_list)
    lengthb = len(backward_prob_list)
    #print length,lengthb
    delta_values_list = []
    for t in range(length-1):
        #print t+1,t+2
        delta_values_at_t = np.dot(np.dot(np.transpose(forward_prob_list[t]),backward_prob_list[t+1])*transition_matrix,observations_inside_list[t+1])
        normalized_delta_values_at_t = delta_values_at_t/probability_of_sequence_given_theta
        #print normalized_delta_values_at_t
        delta_values_list.append(normalized_delta_values_at_t)
    return delta_values_list

def sum_up_all_delta_values(delta_prob_list):
    global final_delta
    for item in delta_prob_list:
        final_delta += item

def update_dict_for_emission_matrix(a_sequence,gamma_prob_list):
    gamma_prob_list = gamma_prob_list[1:]
    for t in range(len(a_sequence)):
        final_dict[dict_of_unique_observations[a_sequence[t]]] += gamma_prob_list[t]

def update_transition_matrix(transition_matrix,final_delta):
    sum = final_delta.sum(axis=1)
    for i in range(number_of_states):
        for j in range(number_of_states):
            transition_matrix[i][j] = final_delta[i][j]/sum[i]
    return transition_matrix

def update_emission_matrix(emission_matrix,final_dict):
    for key,value in final_dict.iteritems():
        for i in range(number_of_states):
            emission_matrix[i,key] = value[i]
    sum = emission_matrix.sum(axis=1)
    for i in range(number_of_states):
        for j in range(number_of_observations):
            emission_matrix[i][j] = emission_matrix[i][j]/sum[i]
    
    return emission_matrix

if __name__ == '__main__':
    
    list_of_observations,dict_of_unique_observations = preprocessing(file_input)
    #print list_of_observations,dict_of_unique_observations
    
    number_of_states = 2
    number_of_observations = len(dict_of_unique_observations)
    mle_previous = -10000.00
    mle_present = -999.00
    
    transition_matrix = np.zeros(shape=(number_of_states,number_of_states))
    emission_matrix = np.zeros(shape=(number_of_states,number_of_observations))
    initial_probabilities = np.zeros(shape=(number_of_states,1))
    endstate_probability = np.zeros(shape=(number_of_states,1))
    
    initial_probabilities = initialize_initial_probabilities(initial_probabilities) 
    endstate_probability = np.ones(shape=(number_of_states,1))    
    transition_matrix = initialize_transition_matrix(transition_matrix)
    emission_matrix = initialize_emission_matrix(emission_matrix)
    #print_trans_emission_matrices(transition_matrix,emission_matrix)

    while(abs(mle_previous-mle_present) > 0.01):
        
        mle_previous = mle_present
        mle_present = 0.0
        
        final_delta = np.zeros(shape=(number_of_states,number_of_states))
        final_dict = {}
        
        for key in dict_of_unique_observations.iterkeys():
            final_dict[dict_of_unique_observations[key]] = np.zeros(shape=(number_of_states,1))
        
        for a_sequence in list_of_observations:
            #print "##"
            #print a_sequence
            matrix_form_a_observation_sequence_inside_list = convert_given_observation_sequence_to_matrix_form(number_of_states,a_sequence,emission_matrix)
            
            forward_prob_list = forward_algorithm(initial_probabilities,transition_matrix,matrix_form_a_observation_sequence_inside_list)
            backward_prob_list = backward_algorithm(endstate_probability,transition_matrix, matrix_form_a_observation_sequence_inside_list)
            
            probability_of_sequence_given_theta = forward_prob_list[-1].sum()
            mle_present += np.log(probability_of_sequence_given_theta)
            
            #print probability_of_sequence_given_theta
        
            #expectation_step
        
            gamma_prob_list = gamma_values_compute(forward_prob_list,backward_prob_list, probability_of_sequence_given_theta)
            delta_prob_list = delta_values_compute(forward_prob_list,backward_prob_list,transition_matrix,matrix_form_a_observation_sequence_inside_list, probability_of_sequence_given_theta)
            #print_gamma_delta_probabilities(gamma_prob_list, delta_prob_list)
            
            sum_up_all_delta_values(delta_prob_list)
            update_dict_for_emission_matrix(a_sequence,gamma_prob_list)
        
        print mle_present
        
        #maximization_step
        
        transition_matrix = update_transition_matrix(transition_matrix,final_delta) 
        emission_matrix = update_emission_matrix(emission_matrix,final_dict)
        initial_probabilities = gamma_prob_list[0]
  
