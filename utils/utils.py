from skimage.io import imread
from skimage.transform import resize
import numpy as np
import glob
import os

def get_mean_std(dataset_path, cache_size = 10000):
    '''
    Computes mean and standard deviation of an image dataset.
    
    Note: Std computation is not optimal, currently computing 
    average of stds. However results should be close. 
    
    dataset_path    : string
                        
                        Path to image dataset.
                    
    cache_size      : int
                        
                        Number of values to store in cache
                        to calculate mean and std at once.
    '''
    # read files
    file_names = glob.iglob(dataset_path)
    
    # init a cache container
    container = np.zeros((3, cache_size))
    
    # init mean and stad. deviation values
    mu = np.zeros((3,))
    std = np.zeros((3,))
    
    # init a counter 
    counter = 1
    
    # use first image to get sizes
    im = imread(next(file_names))
    x_size = im.shape[0]
    y_size = im.shape[1]
        
    # iterate over dataset
    for i,f in enumerate(file_names):
        im = imread(f)
        r = np.sum(im[:,:,0])/(255*x_size*y_size)
        g = np.sum(im[:,:,1])/(255*x_size*y_size)
        b = np.sum(im[:,:,2])/(255*x_size*y_size)
            
        container[:, i % cache_size] = np.array([r,g,b])
        
        if (i + 1) % cache_size == 0:
            mu += np.mean(container, axis = 1)
            std += np.std(container, axis = 1)
            
            #reset container
            container = np.zeros((3, cache_size))
            
            # increment counter
            counter += 1
            
    # scale
    mu = mu / (counter - 1)
    std = std / (counter - 1)
    
    print(mu,std)
    
    return mu, std
    
def print_log(text, file_name = 'Log.txt', ends_with = '\n', display = True):
    '''
    Prints output to the log file.
    
    text        : string or List
                        
                        Output text

    file_name   : string

                        Target log file

    ends_with   : string

                        Ending condition for print func.

    display     : Bool

                        Wheter print to screen or not.
    '''
    
    if display:
        print(text, end = ends_with)

    with open(file_name, "a") as text_file:
        print(text, end = ends_with, file = text_file)

def result_path(file_name = None):
    '''
    
    Creates output folder to save results

    file_name   : string

                    Name of target folder name
    
    '''

    # check root folder
    if not os.path.exists('../gdrive/MyDrive/results/'):
        os.mkdir('../gdrive/MyDrive/results/')

    # find target path
    res_idx = len(os.listdir('../gdrive/MyDrive/results/')) + 1

    if file_name is None:
        target_path = os.path.join('../gdrive/MyDrive/results/', 'Training_' + str(res_idx ) + '/')
    else:
        target_path = os.path.join('../gdrive/MyDrive/results/', 'Training_' + file_name + '/')
    
    # create folder
    os.mkdir(target_path)

    return target_path
