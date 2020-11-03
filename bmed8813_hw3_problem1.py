#! /root/anaconda3/envs/tf/bin/python

'''
Maria Ahmad
BMED 8813
HW 3
Problem 1
'''

import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np


def part_a_datasetSummary():
    # initialize dict of classes
    class_dict = {'N':[],'A':[]}
    # open the reference file
    fileHandle = open("./training2017/REFERENCE.csv","r")
    for line in fileHandle:
        line = line.strip().split(',')
        label = line[0]
        sample_type = line[1]

        # filter by sample type
        if sample_type == 'N':
            class_dict['N'].append(label)
        elif sample_type == 'A':
            class_dict['A'].append(label)
    fileHandle.close

    # print dataset summary
    print('Question 1 Part a: ')
    print('Class\t# of Signals')
    for class_name in class_dict.keys():
        class_size = len(class_dict[class_name])
        print(class_name+'\t'+str(class_size))
    print('\n')
    return class_dict


def part_b_qualityControl(class_dict):
    # calculate length of each signal, and note which signals aren't 9000 in length
    for class_name in class_dict:
        new_signal_list = []
        signal_list = class_dict[class_name]
        for signal in signal_list:
            mat_fname = "./training2017/"+signal+".mat"
            mat_contents = sio.loadmat(mat_fname)
            for key in mat_contents:
                # calculate signal length
                signal_length = len(mat_contents[key][0])
                # filter by signal length
                if signal_length == 9000:
                    new_signal_list.append(signal)
        class_dict[class_name] = new_signal_list
    

    print('Question 1 Part b: ')
    print('Class\t# of Signals')
    for class_name in class_dict:
        class_length = len(class_dict[class_name])
        print(class_name+'\t'+str(class_length))
    print('\n')
    return class_dict


def part_c_divideData(class_dict):
    class_N_length = len(class_dict['N'])
    class_A_length = len(class_dict['A'])
    while class_N_length > class_A_length:
        # print(class_N_length)
        # print(class_A_length)
        class_N_list = class_dict['N']
        class_A_list = class_dict['A']
        random_signal = random.choice(class_N_list)
        # print(random_signal)
        class_N_list.remove(random_signal)
        class_A_list.append(random_signal)
        class_N_length = len(class_N_list)
        class_A_length = len(class_A_list)
        class_dict['N'] = class_N_list
        class_dict['A'] = class_A_list
    
    # print('this is the length of N orginal: ' + str(len(class_dict['N'])))
    # print('This is the length of A original: ' +str(len(class_dict['A'])))

    # initialize lists
    training_N = []
    training_A = []
    validation_N = []
    validation_A = []

    # calculate how many signals make up 80%
    A_80 = round(len(class_dict['A']) * .8)
    A_20 = len(class_dict['A']) - A_80
    N_80 = round(len(class_dict['N']) * .8)
    N_20 = len(class_dict['N']) - A_80

    # print('this is the a_80 value: '+str(A_80))
    # print('this is the n_80 value: '+str(N_80))

    while len(training_N) != N_80:
        random_signal = random.choice(class_dict['N'])
        training_N.append(random_signal)
        class_dict['N'].remove(random_signal)
    validation_N = class_dict['N']

    while len(training_A) != A_80:
        random_signal = random.choice(class_dict['A'])
        training_A.append(random_signal)
        class_dict['A'].remove(random_signal)
    validation_A = class_dict['A']

    # print('this is the length of training_N')
    # print(len(training_N))
    # print('this is the length of training_A')
    # print(len(training_A))
    # print('this is the lenght of validation_N')
    # print(len(validation_N))
    # print('this is the lng of validation_a')
    # print(len(validation_A))

    print('Part 1.c)')
    print('The length of the training dataset for class N is : '+str(len(training_N)))
    print('The length of the training dataset for class A is : '+str(len(training_A)))
    print('The length of the validation dataset for class N is: '+str(len(validation_N)))
    print('The length of the validation dataset for class A is: '+str(len(validation_A)))
    print('\n')
    return training_N, validation_N, training_A, validation_A


def part_c_divideData_version2(class_dict):
    # Length of N class and A class
    class_N_length = len(class_dict['N'])
    class_A_length = len(class_dict['A'])

    number_to_lose = class_N_length - class_A_length

    # Drop signals in N until size of class N is equal to the size of class A
    for count in range(0,number_to_lose):
        class_N_list = class_dict['N']
        class_A_list = class_dict['A']
        # random signal in the class N
        random_signal = random.choice(class_N_list)
        # remove the random signal from class N
        class_N_list.remove(random_signal)
        # update the classes in the dictionary
        class_dict['N'] = class_N_list
        class_dict['A'] = class_A_list
    
    # N and A should be the same list size
    print('Question 1 Part c: ')
    print('Class\t# of Signals')
    print('N\t'+str(len(class_dict['N'])))
    print('A\t'+str(len(class_dict['A'])))

    # Adds both N and A class to one list
    total = []
    label_total = []
    for signal in class_dict['N']:
        total.append(signal)
        # 0 is the label for N
        label_total.append('0')
    for signal in class_dict['A']:
        total.append(signal)
        # 1 is the label for A
        label_total.append('1')

    # initialize lists
    training = []
    training_labels = []
    validation = []
    validation_labels = []

    total_len = len(total) # first half is N, second half is A
    total_80 = round(total_len * .8)
    total_20 = total_len - total_80
    
    # Add 80 percent of the data to the training set, and the rest for the validation set
    for count in range(0,total_80):
        # choses a random index in the total dataset
        random_index = random.randint(0,total_len-1)
        # captures the element at that index in the total dataset
        random_signal = total[random_index]
        # captures the label for the element at that index in the total dataset
        random_signal_label = label_total[random_index]
        # adds the element to the training dataset
        training.append(random_signal)
        # adds the label to the labels datset
        training_labels.append(random_signal_label)
        # deletes the element and label from the total and og label datasets, so that they aren't picked again
        del total[random_index]
        del label_total[random_index]
        total_len = len(total)

    # 80 percent is the training set, so the remaining total dataset becomes the validation dataset
    validation = total

    print('Dataset\tSize')
    print('Training\t'+str(len(training)))
    print('Validation\t'+str(len(validation)))
    print('\n')

    return training, training_labels, validation, validation_labels


def prepare_data_version2(training, training_labels, validation, validation_labels):
    # initialize lists
    training_data = []
    validation_data = []

    # previous lists only had the ID for the signal, this 
    # replaces the ID with the actual array
    for signalName in training:
        mat_fname = "./training2017/"+signalName+".mat"
        mat_contents = sio.loadmat(mat_fname)
        # mat_contents is a dictionary
        # 'val' is the key to the array
        # adds the array to the training_data
        training_data.append(mat_contents['val'][0])
    for signalName in validation:
        mat_fname = "./training2017/"+signalName+".mat"
        mat_contents = sio.loadmat(mat_fname)
        validation_data.append(mat_contents['val'][0])
    # print(validation_data)
    # print(mat_contents)
    # print(type(validation_data[0]))
    # print(validation_data[0][0])
    return training_data, training_labels, validation_data, validation_labels


def prepare_data(training_N, validation_N, training_A, validation_A):
    tN = []
    vN = []
    tA = []
    vA = []
    for signalName in training_N:
        mat_fname = "./training2017/"+signalName+".mat"
        mat_contents = sio.loadmat(mat_fname)
        tN.append(mat_contents)
    for signalName in training_A:
        mat_fname = "./training2017/"+signalName+".mat"
        mat_contents = sio.loadmat(mat_fname)
        tA.append(mat_contents)
    for signalName in validation_N:
        mat_fname = "./training2017/"+signalName+".mat"
        mat_contents = sio.loadmat(mat_fname)
        vN.append(mat_contents)
    for signalName in validation_A:
        mat_fname = "./training2017/"+signalName+".mat"
        mat_contents = sio.loadmat(mat_fname)
        vA.append(mat_contents)
    return tN, tA, vN, vA


def part_d_createNetwork(training_data, training_labels, validation_data, validation_labels):
    '''Input layer(s), as needed '''

    print('Question 1 Part d: ')


    epochs_number = 10
    sequence_length = 1000
    initial_learn_rate = 0.01
    gradient_threshold = 1
    execution_environment = 'auto'
    verbose = True

    # Converts the lists to arrays
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    validation_data = np.array(validation_data)
    validation_labels = np.array(validation_labels)


    ##### Shape of training data is (798, 9000)
    ##### Shape of training labels data is (798,)


    # Converts the lists to tensors
    # training_data = tf.convert_to_tensor(training_data)
    # training_data = tf.convert_to_tensor(training_labels)
    # training_data = tf.convert_to_tensor(validation_data)
    # training_data = tf.convert_to_tensor(validation_labels)

    # 9000 samples in the signal
    data_dim = 9000
    time_step = len(training_data)
    the_batch_size = 150
    try:
        print('Step 1')
        model = tf.keras.models.Sequential()
        print('Adding LSTM layer')

        ######## most simple
        # model.add(tf.keras.layers.LSTM(100))

        ######## what i THOUGHT the indian youtuber was saying
        # model.add(tf.keras.layers.LSTM(100,batch_input_shape=(the_batch_size,time_step,data_dim),return_sequences=False))

        ######## other version of indian youtuber
        model.add(tf.keras.layers.LSTM(100,batch_input_shape=(time_step,1,data_dim),return_sequences=False))

        ######## The sentdex guy's line
        # model.add(tf.keras.layers.LSTM(100,input_shape=training_data.shape,activation='relu',return_sequences=False))
        
        ######## what a website said to do
        # model.add(tf.keras.layers.LSTM(100,batch_input_shape=(None,time_step,data_dim),return_sequences=False))
        
        print('Compiling')
        model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
        print('Summarizing')
        model.summary()
        print('Fitting')

        ### indian youtuber's line
        model.fit(training_data,training_labels,batch_size=the_batch_size,epochs=epochs_number,validation_data=(validation_data,validation_labels))
    except:
        print('** Fitting the model failed. Data type error **')
    
    ### the sentdex guy's line
    # model.fit(training_data,training_labels,epochs=epochs_number,validation_data=(validation_data,validation_labels))
    
    ### indian youtuber's line, without the validation data
    # model.fit(training_data,training_labels,batch_size=the_batch_size,epochs=epochs_number)#,validation_data=(validation_data,validation_labels))
    
    ################### old is not gold
    # model.add(tf.keras.layers.LSTM(1,batch_size=the_batch_size),batch_input_shape(the_batch_size,1000,9000))
    # model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
    # # model.summary()
    # model.fit(tN,tA,epochs=the_epochs,validation_data=(vN,vA),verbose=2,batch_size=the_batch_size)
    # # number of inputs, length of input sequences, and length of each vector
    return True


def part_e_confusionMatrix():
    print('Question 1 Part e: ')
    print('* LSTM did not work * ')
    return 

def main():
    class_dict = part_a_datasetSummary()
    class_dict = part_b_qualityControl(class_dict)
    training, training_labels, validation, validation_labels = part_c_divideData_version2(class_dict)
    training_data, training_labels, validation_data, validation_labels = prepare_data_version2(training, training_labels, validation, validation_labels)
    part_d_createNetwork(training_data, training_labels, validation_data, validation_labels)
    part_e_confusionMatrix()
    return True


################
main()
################
