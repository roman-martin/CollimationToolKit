# CollimationToolKit
Provides classes and routines for collimation studies with pysixtrack.

The current state of the package is experimental and it's structure can still
change significantly. Even the scope of this package might change as pysixtrack
functionality develops.

The code tries to stick to the pysixtrack conventions, but will circumvent them
where it is necessary in order to provide the intended functionality.

This package is meant to provide elements that include USER-PROVIDED functions
to simulate particle-matter interaction. However, the package will provide some
exemplary interfaces to specific particle-matter simulation codes. The limits
of these codes and of the interfaces should be understood before using them.


## Prerequisites
The package is based on the pysixtrack tracking engine.
```
pip install -U pysixtrack
```

To make use of the GLOBAL charge exchange code, you need to run on a Linux
system. You need to download the Linux executable of GLOBAL from 
https://web-docs.gsi.de/~weick/charge_states/ and make sure it is in your path
```
export PATH="/path/to/global/directory:$PATH"
```

## Running the tests

The package's tests depend on pytest. After installation, run the tests via:
```
python -m pytest
```
or simply
```
pytest
```
