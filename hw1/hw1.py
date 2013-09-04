#!/usr/bin/python2

import numpy as np
import dist

def main():
    # Initialize data
    # dt is a data container containing rows/labels
    
    dt = np.dtype([('labels',np.str_, 1), ('data',np.int16, (16,))])
    
    testData = np.zeros((5000,1),dt)
    trainData = np.zeros((15000,1),dt)
    
    counter = 0
    
    # Read in data
    
    with open('letter-recognition.data') as f:
        for line in f:
            content = line.split(',')
            
            label = content.pop(0)
            data = np.int16(content)
            
            if counter < 15000:
                trainData.put(counter,[(label,data)])
            else:
                testData.put(counter-15000,[(label,data)])
            
            counter = counter + 1
         # end for line in f
    # end with open()
    
    
        

if __name__ == "__main__":
    main()