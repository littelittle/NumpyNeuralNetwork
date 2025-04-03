import numpy as np

def Xavier_init(size:tuple):
    """
    if the initialization follows normal distribution, 
    the Xavier way is to let the var to be 2/(fan_in + fan_out)
    enabling the var(output[*]) both forward and backward to near 1
    """

    if len(size) == 2:
        # the most common matrix multi
        stddev = np.sqrt(2/(sum(size)))
        return np.random.normal(size=size)*stddev
    elif len(size) == 4:
        # something like kernel in conv2d (out, in, k, k)
        stddev = np.sqrt(2/(sum(size)+sum(size[-2:]))) # which means out+k+k & in+k+k
        return np.random.normal(size=size)*stddev
    elif len(size) == 1:
        return np.random.normal(size=size)/10 # this is just bias
    else:
        raise NotImplementedError(f"size length {len(size)} has not been implemented yet!")
    

def Kaiming_init(size:tuple):
    """
    simular to Xavier, but for relu(which will deactivate 1/2 nuerons at begining)
    """

    if len(size) == 2:
        # the most common matrix multi
        stddev = np.sqrt(2/(size[0])) # 2/fan_in 
        return np.random.normal(size=size)*stddev
    elif len(size) == 3:
        # something like kernel in conv2d (out, in, k, k)
        stddev = np.sqrt(2/(sum(size[1:]))) # which means in+k+k
        return np.random.normal(size=size)*stddev
    elif len(size) == 1:
        return np.random.normal(size=size)/10 # this is just bias
    else:
        raise NotImplementedError(f"size length {len(size)} has not been implemented yet!")