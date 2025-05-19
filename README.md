# GNC Tools
This is a Python package for getting easy access to tools needed to perform calculations and transformation related to guidance, navigation and control (GNC) tasks. It is based on the tools used in [PythonVehicleSimulator](https://github.com/cybergalactic/PythonVehicleSimulator/) and [MSS Toolbox](https://github.com/cybergalactic/MSS) by Thor I. Fossen.

## Installation
### pip install
Coming soon

### Install from source
 

## Usage
### GNC functions
```python
import numpy as np
import gnc

nu = np.array([2, 0, 0])
eta_dot = gnc.B2N(nu) # BODY to NED transformation
```

### Linear algebra
```python
import numpy as np
import gnc

A = np.array([
    [2, 0, 0],
    [1, 0, 3],
    [9, 0, 4]

]) # Non-invertable matrix

# Inverting using Moore-Penrose pseudo-inverse
A_inv = gnc.linalg.moore_penrose(A)
```