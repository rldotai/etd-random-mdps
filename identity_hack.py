import numpy as np 

class Identity:
    """Instantiate an object that's always the identity matrix!"""
    def __add__(self, other):
        return np.eye(len(other)).__add__(other)
    
    def __mul__(self, other):
        return np.eye(len(other)).__mul__(other)
    
    def __sub__(self, other):
        return np.eye(len(other)).__sub__(other)