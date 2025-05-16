# share numpy array between processes via a shared raw array
from multiprocessing import Process
from multiprocessing.sharedctypes import RawArray
from numpy import frombuffer
from numpy import double
import numpy as np
import time

# task executed in a child process
def main_task(array):
    data = frombuffer(array, dtype=double, count=len(array))
    # create a new numpy array backed by the raw array
    for i in range(100):
        np.copyto(data[i:i+1], i/1000.0)
        # print(array[i])
        time.sleep(0.001)


def task(array):
    data = frombuffer(array, dtype=double, count=len(array))

    # create a new numpy array backed by the raw array
    for i in range(100):
        print(data[i])
        time.sleep(0.1)
 
# protect the entry point
if __name__ == '__main__':
    # define the size of the numpy array
    n = 10000000
    # create the shared array
    array = RawArray('d', n)
    # create a new numpy array backed by the raw array
    data = frombuffer(array, dtype=double, count=len(array))
    # create a child process
    main = Process(target=main_task, args=(array,))
    child = Process(target=task, args=(array,))
    # start the child process
    main.start()
    child.start()
    # wait for the child process to complete
    main.join()
    child.join()
    # check some data in the shared array