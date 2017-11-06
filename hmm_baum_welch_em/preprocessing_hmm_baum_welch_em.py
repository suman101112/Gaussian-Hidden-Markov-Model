import numpy as np
list_of_observations = []
set_of_unique_observations=set()
dict_of_unique_observations={}

def preprocessing_to_extract_observations(line):
    if(len(line)>0):
        list = line.replace("\n","").split(",")
        list_of_observations.append(list)
        set_of_unique_observations.update(list)

def preprocessing_to_map_observations_to_column_numbers(set_of_unique_observations,list_of_observations):
    count = 0
    for item in set_of_unique_observations:
        dict_of_unique_observations[item] = count
        count += 1
    print dict_of_unique_observations
    
def preprocessing(file_input):
    for line in file_input:
        preprocessing_to_extract_observations(line)
    preprocessing_to_map_observations_to_column_numbers(set_of_unique_observations,list_of_observations)
        
    return list_of_observations,dict_of_unique_observations

def convert_given_observation_sequence_to_matrix_form(number_of_states,sequence,emission_matrix):
    list = []
    for item in sequence:
        observation_matrix = np.zeros(shape=(number_of_states,number_of_states))
        observations = emission_matrix[:,dict_of_unique_observations[item]]
        for i in range(number_of_states):
            observation_matrix[i][i] = observations[i]
        list.append(observation_matrix)
        #print observation_matrix
    #print list        
    return list
