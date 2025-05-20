# GNC Tools
This is a Python package for getting easy access to tools needed to perform calculations and transformation related to guidance, navigation and control (GNC) tasks. It is based on the tools used in [PythonVehicleSimulator](https://github.com/cybergalactic/PythonVehicleSimulator/) and [MSS Toolbox](https://github.com/cybergalactic/MSS) by Thor I. Fossen.

## Installation
### pip install
Coming soon

### Install from source
1. Clone repo using ```git clone git@github.com:rambech/gnc-tools.git```
2. Enter the repo using ```cd gnc-tools```
3. Install package using ```python3 -m pip install .```

## Usage
### Import
```python
import gnc # Main library
import gnc.linalg as linalg # Linear algebra library
```

### GNC functions
```python
import numpy as np
import gnc

# Transforms
nu = np.array([2, 0, 0])
eta_dot = gnc.B2N(nu) # BODY to NED transformation

# Helpful functions
wp0 = [59.65918355434504, 10.62844055814729]
wp1 = [59.90875817046556, 10.71927031763238]
distance = gnc.distance_along_great_circle(wp0[0], wp0[1], wp1[0], wp[1])
```

### Linear algebra
```python
import numpy as np
import gnc

A = np.array([
    [2, 0, 0],
    [1, 0, 3],
    [7, 0, 5]

]) # Non-invertable matrix

# Inverting using Moore-Penrose pseudo-inverse
A_inv = gnc.linalg.moore_penrose(A)

# 3x3 skew-symmetric matrix
vector = np.array([1, 5, 3])
S = gnc.linalg.Smtrx(vector)
```

### Mathematics shortcuts
```python
import numpy as np
import gnc

# Conversion
radians = 2*np.pi
degrees = gnc.R2D(radians)
ragians = gnc.D2R(degrees)

speed_knots = 11
speed_ms = gnc.kts2ms(speed_knots)
speed_knots = gnc.ms2kts(speed_ms)
```


### Full function list
A full list of available functions and their description will be available at some point.