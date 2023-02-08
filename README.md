# fast-mean-shift
This is a C++ implemenation of the sklearn MeanShift algorithm 
(https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/cluster/_mean_shift.py),
with Eigen3 and OpenMP acceleration, and Python API provided via Pybind11. Comparing to sklearn, this implementation is much more efficient on high-dimensional data (>100D). But a larger memory space is required as the implementation generates large intermediate matrices. 

### 1. Requirements
To compile the library, please make sure you have GCC & G++ installed, and the support for C++ 11 is required.

### 2. Installation
First install the Pybind11 library:
```
pip install pybind11
```
Then compile the c++ backend and the python API:
```
make fast_mean_shift
```
Finally copy the generated library file (located in the directory `lib/`) to your python package installation directory. You can use this command to check the package directory:
```
python -m site
```

### 3. APIs
Two major APIs:

(1) `mean_shift_clustering`: accepting the same parameter list as `sklearn.cluster.MeanShift` in sklean,  returning the centroids (shape (n_centroids, n_features)) and the cluster IDs for all data points (shape (n_points,)).

(2) `mean_shift_density_peaks_detecting`: accepting the same parameter list as `mean_shift_clustering`, returning the centroids (also the density peaks) found by mean shift algorithm, whose shape is (n_centroids, n_features).

Two minor APIs:

(1) `estimate_bandwidth`: accepting the same parameter list as `sklearn.cluster.estimate_bandwidth` in sklean, estimating and returning the proper bandwidth.

(2) `setDistanceMetric`: setting the distance metric used in FastMeanShift, supported metrics are `Metric.DIST_L2` for L2 distance and `Metric.DIST_COS` for cosine distance.


### 4. Demo
Please see `test_fast_mean_shift.ipynb` for a demo that compares `sklearn.cluster.MeanShift` and `fast_mean_shift.FastMeanShift`. 

### 5. Developer
Most of the functions in fast_mean_shift.cpp are translated from those in sklearn, please refer to file https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/cluster/_mean_shift.py for detailed comments. 

### Note:
(1) The APIs only accept numpy arrays with float32 data type. To make it accept float64, please change the definition of `VectorElemType` in `util.h` from `float` to `double`. 

