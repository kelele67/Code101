
import numpy as np
print(np.__version__)

arr = np.arange(10)
print(arr)

arr = np.full((3, 3), True, dtype=bool)
print(arr)
arr = np.ones((3, 3), dtype=bool)
print(arr)

# Input
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Sol
arr = arr[arr % 2 == 1]
print(arr)

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr[arr % 2 == 1] = -1
print(arr)

arr = np.arange(10)
out = np.where(arr % 2 == 1, -1, arr)
print(arr)
print(out)

arr = np.arange(10)
arr = arr.reshape(2, -1) # Setting to -1 automatically decides the number of cols
print(arr)

a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)
# Method 1
arr = np.concatenate([a, b], axis=0)
print(arr)
# Method 2
arr = np.vstack([a, b])
print(arr)
# Method 3
arr = np.r_[a, b]
print(arr)

a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)
# Method 1
arr = np.concatenate([a, b], axis=1)
print(arr)
# Method 2
arr = np.hstack([a, b])
print(arr)
# Method 3
arr = np.c_[a, b]
print(arr)

a = np.array([1, 2, 3])
arr = np.r_[np.repeat(a, 3), np.tile(a, 3)]
print(arr)

a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
arr = np.intersect1d(a, b)
print(arr)

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])
# From 'a' remove all of 'b'
arr = np.setdiff1d(a, b)
print(arr)

a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
arr = np.where(a == b)
print(arr)

a = np.arange(15)
# Method 1
index = np.where((a >= 5) & (a <= 10))
print(a[index])
# Method 2
index = np.where(np.logical_and(a>=5, a<=10))
print(a[index])
# Method 3
arr = a[(a >= 5) & (a <= 10)]
print(arr)

def maxx(x, y):
    if x >= y:
        return x
    else:
        return y
pair_max = np.vectorize(maxx, otypes=[float])
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
arr = pair_max(a, b)
print(arr)

arr = np.arange(9).reshape(3, 3)
print(arr)
arr = arr[:, [1, 0, 2]]
print(arr)

arr = np.arange(9).reshape(3, 3)
print(arr)
arr = arr[::-1]
print(arr)    

arr = np.arange(9).reshape(3, 3)
print(arr)
arr = arr[:, ::-1]
print(arr)

# Input
arr = np.arange(9).reshape(3, 3)
# Sol1
rand_arr = np.random.randint(low=5, high=10, size=(5, 3)) + np.random.random((5, 3))
print(rand_arr)
# Sol2
rand_arr = np.random.uniform(5, 10, size=(5, 3))
print(rand_arr)

rand_arr = np.random.random([5, 3]) 
# Limit to 3 decimal places
np.set_printoptions(precision=3)
rand_arr = rand_arr[:4]
print(rand_arr)

np.random.seed(100)
rand_arr = np.random.random([3, 3]) / 1e3
print(rand_arr)   

np.set_printoptions(threshold=6)
a = np.arange(15)
print(a)

np.set_printoptions(threshold=np.nan)
print(a)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
print(iris[:3])

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
# Method1
iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
print(iris_2d[:4])
# Method2
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
print(iris_2d[:4])

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
# Sol
mu, med, sd = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
print(mu, med, sd)

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
# Sol
Smax, Smin = sepallength.max(), sepallength.min()
S = (sepallength - Smin) / (Smax - Smin)
print(S)
# or
S = (sepallength - Smin) / sepallength.ptp()
print(S)

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.array([float(row[0]) for row in iris])
# Sol
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
print(softmax(sepallength))

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
# Sol
arr = np.percentile(sepallength, q=[5, 95])
print(arr)

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
# Method 1
i, j = np.where(iris_2d)
np.random.seed(100)
iris_2d[np.random.choice((i), 20), np.random.choice((j), 20)] = np.nan
print(iris_2d[:10])
# Method 2
np.random.seed(100)
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
print(iris_2d[:10])

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
# Sol
print("Number of missing values: \n", np.isnan(iris_2d[:, 0]).sum())
print("Position of missing values: \n", np.where(np.isnan(iris_2d[:, 0])))

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
# Sol
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
print(iris_2d[condition])

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
# Sol
# Method 1
any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])
print(iris_2d[any_nan_in_row][:5])
# Method 2
print(iris_2d[np.sum(np.isnan(iris_2d), axis=1) == 0][:5])

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
# Method 1
print(np.corrcoef(iris[:, 0], iris[:, 2])[0, 1])
# Method 2
from scipy.stats.stats import pearsonr
corr, p_value = pearsonr(iris[:, 0], iris[:, 2])
print(corr)

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
print(np.isnan(iris_2d).any())

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
# Sol
iris_2d[np.isnan(iris_2d)] = 0
print(iris_2d[:4])

# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
# Sol
species = np.array([row.tolist()[4] for row in iris])
print(np.unique(species, return_counts=True))

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
# Bin petallength
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])
# Map it to respective category
label_map = {1:'small', 2:'medium', 3:'large', 4:np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]
# View
print(petal_length_cat[:4])

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
# Sol
# Compute volume
sepallength = iris_2d[:, 0].astype('float')
petallength = iris_2d[:, 2].astype('float')
volume = (np.pi * petallength * (sepallength**2)) / 3
# Introduce new dimension to match iris_2d's
volume = volume[:, np.newaxis]
# Add the new column
out = np.hstack([iris_2d, volume])
# View
print(out[:4])

# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
# Sol
# Get the species column
species = iris[:, 4]
# Generate Probablistically
np.random.seed(100)
a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
secies_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])
# Probablistic Sampling (preferred)
np.random.seed(100)
probs = np.r_[np.linspace(0, 0.500, num=50), np.linspace(0.501, .750, num= 50), np.linspace(.751, 1.0, num=50)]
index = np.searchsorted(probs, np.random.random(150))
species_out = species[index]
print(np.unique(species_out, return_counts=True))

# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
# Sol
# Get the species and petal length columns
petal_len_setosa = iris[iris[:, 4] == b'Iris-setosa', [2]].astype('float')
# Get the second last value
print(np.unique(np.sort(petal_len_setosa))[-2])

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
# Sol
print(iris[iris[:, 0].argsort()][:20])

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
# Sol
vals, counts = np.unique(iris[:, 2], return_counts=True)
print(vals[np.argmax(counts)])

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
# Sol
print(np.argwhere(iris[:, 3].astype(float) > 1.0)[0])

# Input
np.set_printoptions(precision=2)
np.random.seed(100)
a = np.random.uniform(1,50, 20)
# Method1
print(np.clip(a, a_min=10, a_max=30))
# Method2
print(np.where(a < 10, 10, np.where(a > 30, 30, a)))

# Input
np.random.seed(100)
a = np.random.uniform(1,50, 20)
# Sol1
print(a.argsort())
# Sol2
print(np.argpartition(-a, 5)[:5])
# Method1
print(a[a.argsort()][-5:])
# Method2
print(np.sort(a)[-5:])
# Method3
print(np.partition(a, kth=-5)[-5:])
# Method4
print(a[np.argpartition(-a, 5)][:5])

# Input
np.random.seed(100)
arr = np.random.randint(1,11,size=(6, 10))
print(arr)
# Sol
def counts_of_all_values_rowwise(arr2d):
    # Unique values and its counts row wise
    num_counts_array = [np.unique(row, return_counts=True) for row in arr2d]
    # Counts of all values row wise
    return([[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in num_counts_array])
# Print
print(np.arange(1, 11))
print(counts_of_all_values_rowwise(arr))    
# Example 2:
arr = np.array([np.array(list('bill clinton')), np.array(list('narendramodi')), np.array(list('jjayalalitha'))])
print(np.unique(arr))
print(counts_of_all_values_rowwise(arr))

# Input
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)
array_of_arrays = np.array([arr1, arr2, arr3])
print('array_of_arrays: ', array_of_arrays)
# Method1
arr_2d = np.array([a for arr in array_of_arrays for a in arr])
print(arr_2d)
# Method2
arr_2d = np.concatenate(array_of_arrays)
print(arr_2d)

# Input
np.random.seed(101) 
arr = np.random.randint(1,4, size=6)
print(arr)
# Sol
def ont_hot_encodings(arr):
    uniqs = np.unique(arr)
    out = np.zeros((arr.shape[0], uniqs.shape[0]))
    for i, k in enumerate(arr):
        out[i, k-1] = 1
    return out
print(ont_hot_encodings(arr))
# Method2
print((arr[:, None] == np.unique(arr)).view(np.int8))

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
np.random.seed(100)
# Sol
species_small = np.sort(np.random.choice(species, size=20))
print(species_small)

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
np.random.seed(100)
species_small = np.sort(np.random.choice(species, size=20))
print(species_small)
# Sol
output = [np.argwhere(np.unique(species_small) == s).tolist()[0][0] for val in np.unique(species_small) for s in species_small[species_small==val]]
print(output)
# Sol for loop version
output = []
uniqs = np.unique(species_small)
for val in uniqs:
    for s in species_small[species_small == val]:
        groupid = np.argwhere(uniqs == s).tolist()[0][0]
        output.append(groupid)
print(output)

# Input
np.random.seed(10)
a = np.random.randint(20, size=10)
print('Array: ', a)
# Sol
print(a.argsort().argsort())

# Input
np.random.seed(10)
a = np.random.randint(20, size=[2,5])
print(a)
# Sol
print(a.ravel().argsort().argsort().reshape(a.shape))

# Input
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
print(a)
# Method1
print(np.amax(a, axis=1))
# Method2
print(np.apply_along_axis(np.max, arr=a, axis=1))

# Input
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
print(a)
# Sol
print(np.apply_along_axis(lambda x: np.min(x) / np.max(x), arr=a, axis=1))

# Input
np.random.seed(100)
a = np.random.randint(0, 5, 10)
print(a)
# Sol
out = np.full(a.shape[0], True)
unique_positions = np.unique(a, return_index=True)[1]
out[unique_positions] = False
print(out)

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
# Sol
numeric_column = iris[:, 1].astype('float')
grouping_column = iris[:, 4]
print([[group_val, numeric_column[grouping_column==group_val].mean()] for group_val in np.unique(grouping_column)])
# Sol for loop version
output = []
for group_val in np.unique(grouping_column):
    output.append([group_val, numeric_column[grouping_column==group_val].mean()])
print(output)

from io import BytesIO
from PIL import Image
import PIL, requests
# Import image from URL
URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
response = requests.get(URL)
I = Image.open(BytesIO(response.content))
I = I.resize([150, 150])
arr = np.asarray(I)
im = PIL.Image.fromarray(np.uint8(arr))
# Image.Image.show(im)

# Input
a = np.array([1,2,3,np.nan,5,6,7,np.nan])
# Sol
print(a[~np.isnan(a)])

# Input
a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])
# Sol
dist = np.linalg.norm(a-b)
print(dist)

# Input
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
doublediff = np.diff(np.sign(np.diff(a)))
peak_locations = np.where(doublediff == -2)[0] + 1
print(peak_locations)

# Input
a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
b_1d = np.array([1,2,3])
# Sol
print(a_2d - b_1d[:, None])

# Input
x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
n = 5
# Method1
print([i for i, v in enumerate(x) if v == 1][n-1])
# Method2
print(np.where(x == 1)[0][n-1])

# Input
dt64 = np.datetime64('2018-02-25 22:10:10')
# Sol
from datetime import datetime
print(dt64.tolist())
print(dt64.astype(datetime))

# Input
np.random.seed(100)
Z = np.random.randint(10, size=10)
print('array: ', Z)
# Sol
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
np.random.seed(100)
# Method1
print(moving_average(Z, n=3).round(2))
# Method2
print(np.convolve(Z, np.ones(3)/3, mode='valid'))

length = 10
start = 5
step = 3
def seq(start, length, step):
    end = start + (step*length)
    return np.arange(start, end, step)
print(seq(start, length, step))

# Input
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
print(dates)
# Sol
filled_in = np.array([np.arange(date, (date+d)) for date, d in zip(dates, np.diff(dates))]).reshape(-1)
# add the last day
output = np.hstack([filled_in, dates[-1]])
print(output)
# Sol loop version
out = []
for date, d in zip(dates, np.diff(dates)):
    out.append(np.arange(date, (date+d)))
filled_in = np.array(out).reshape(-1)
# add the last day
output = np.hstack([filled_in, dates[-1]])
print(output)

def gen_strides(a, stride_len=5, window_len=5):
    n_strides = ((a.size - window_len) // stride_len) + 1
    return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides * stride_len, stride_len)])
print(gen_strides(np.arange(15), stride_len=2, window_len=4))

"""
output
1.12.1
[0 1 2 3 4 5 6 7 8 9]
[[ True  True  True]
 [ True  True  True]
 [ True  True  True]]
[[ True  True  True]
 [ True  True  True]
 [ True  True  True]]
[1 3 5 7 9]
[ 0 -1  2 -1  4 -1  6 -1  8 -1]
[0 1 2 3 4 5 6 7 8 9]
[ 0 -1  2 -1  4 -1  6 -1  8 -1]
[[0 1 2 3 4]
 [5 6 7 8 9]]
[[0 1 2 3 4]
 [5 6 7 8 9]
 [1 1 1 1 1]
 [1 1 1 1 1]]
[[0 1 2 3 4]
 [5 6 7 8 9]
 [1 1 1 1 1]
 [1 1 1 1 1]]
[[0 1 2 3 4]
 [5 6 7 8 9]
 [1 1 1 1 1]
 [1 1 1 1 1]]
[[0 1 2 3 4 1 1 1 1 1]
 [5 6 7 8 9 1 1 1 1 1]]
[[0 1 2 3 4 1 1 1 1 1]
 [5 6 7 8 9 1 1 1 1 1]]
[[0 1 2 3 4 1 1 1 1 1]
 [5 6 7 8 9 1 1 1 1 1]]
[1 1 1 2 2 2 3 3 3 1 2 3 1 2 3 1 2 3]
[2 4]
[1 2 3 4]
(array([1, 3, 5, 7]),)
[ 5  6  7  8  9 10]
[ 5  6  7  8  9 10]
[ 5  6  7  8  9 10]
[ 6.  7.  9.  8.  9.  7.  5.]
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[1 0 2]
 [4 3 5]
 [7 6 8]]
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[6 7 8]
 [3 4 5]
 [0 1 2]]
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[2 1 0]
 [5 4 3]
 [8 7 6]]
[[ 9.65394107  8.17929591  9.13968255]
 [ 8.20815608  9.04141335  9.34020519]
 [ 9.7115973   7.15676274  8.55559863]
 [ 8.52900696  9.5506496   8.94620623]
 [ 9.12598615  6.9792719   7.5646151 ]]
[[ 9.18020728  9.97548009  6.51388629]
 [ 7.92908095  6.73351751  6.18845085]
 [ 9.99705183  7.99633563  7.87563736]
 [ 8.12790572  7.12392818  7.22112169]
 [ 5.29151515  6.04021952  6.62095235]]
[[ 0.783  0.569  0.197]
 [ 0.318  0.347  0.643]
 [ 0.192  0.931  0.688]
 [ 0.373  0.322  0.603]]
[[  5.434e-04   2.784e-04   4.245e-04]
 [  8.448e-04   4.719e-06   1.216e-04]
 [  6.707e-04   8.259e-04   1.367e-04]]
[ 0  1  2 ..., 12 13 14]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
[[b'5.1' b'3.5' b'1.4' b'0.2' b'Iris-setosa']
 [b'4.9' b'3.0' b'1.4' b'0.2' b'Iris-setosa']
 [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa']]
[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]
 [ 4.7  3.2  1.3  0.2]
 [ 4.6  3.1  1.5  0.2]]
[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]
 [ 4.7  3.2  1.3  0.2]
 [ 4.6  3.1  1.5  0.2]]
5.84333333333 5.8 0.825301291785
[ 0.222  0.167  0.111  0.083  0.194  0.306  0.083  0.194  0.028  0.167
  0.306  0.139  0.139  0.     0.417  0.389  0.306  0.222  0.389  0.222
  0.306  0.222  0.083  0.222  0.139  0.194  0.194  0.25   0.25   0.111
  0.139  0.306  0.25   0.333  0.167  0.194  0.333  0.167  0.028  0.222
  0.194  0.056  0.028  0.194  0.222  0.139  0.222  0.083  0.278  0.194
  0.75   0.583  0.722  0.333  0.611  0.389  0.556  0.167  0.639  0.25
  0.194  0.444  0.472  0.5    0.361  0.667  0.361  0.417  0.528  0.361
  0.444  0.5    0.556  0.5    0.583  0.639  0.694  0.667  0.472  0.389
  0.333  0.333  0.417  0.472  0.306  0.472  0.667  0.556  0.361  0.333
  0.333  0.5    0.417  0.194  0.361  0.389  0.389  0.528  0.222  0.389
  0.556  0.417  0.778  0.556  0.611  0.917  0.167  0.833  0.667  0.806
  0.611  0.583  0.694  0.389  0.417  0.583  0.611  0.944  0.944  0.472
  0.722  0.361  0.944  0.556  0.667  0.806  0.528  0.5    0.583  0.806
  0.861  1.     0.583  0.556  0.5    0.944  0.556  0.583  0.472  0.722
  0.667  0.722  0.417  0.694  0.667  0.667  0.556  0.611  0.528  0.444]
[ 0.222  0.167  0.111  0.083  0.194  0.306  0.083  0.194  0.028  0.167
  0.306  0.139  0.139  0.     0.417  0.389  0.306  0.222  0.389  0.222
  0.306  0.222  0.083  0.222  0.139  0.194  0.194  0.25   0.25   0.111
  0.139  0.306  0.25   0.333  0.167  0.194  0.333  0.167  0.028  0.222
  0.194  0.056  0.028  0.194  0.222  0.139  0.222  0.083  0.278  0.194
  0.75   0.583  0.722  0.333  0.611  0.389  0.556  0.167  0.639  0.25
  0.194  0.444  0.472  0.5    0.361  0.667  0.361  0.417  0.528  0.361
  0.444  0.5    0.556  0.5    0.583  0.639  0.694  0.667  0.472  0.389
  0.333  0.333  0.417  0.472  0.306  0.472  0.667  0.556  0.361  0.333
  0.333  0.5    0.417  0.194  0.361  0.389  0.389  0.528  0.222  0.389
  0.556  0.417  0.778  0.556  0.611  0.917  0.167  0.833  0.667  0.806
  0.611  0.583  0.694  0.389  0.417  0.583  0.611  0.944  0.944  0.472
  0.722  0.361  0.944  0.556  0.667  0.806  0.528  0.5    0.583  0.806
  0.861  1.     0.583  0.556  0.5    0.944  0.556  0.583  0.472  0.722
  0.667  0.722  0.417  0.694  0.667  0.667  0.556  0.611  0.528  0.444]
[ 0.002  0.002  0.001  0.001  0.002  0.003  0.001  0.002  0.001  0.002
  0.003  0.002  0.002  0.001  0.004  0.004  0.003  0.002  0.004  0.002
  0.003  0.002  0.001  0.002  0.002  0.002  0.002  0.002  0.002  0.001
  0.002  0.003  0.002  0.003  0.002  0.002  0.003  0.002  0.001  0.002
  0.002  0.001  0.001  0.002  0.002  0.002  0.002  0.001  0.003  0.002
  0.015  0.008  0.013  0.003  0.009  0.004  0.007  0.002  0.01   0.002
  0.002  0.005  0.005  0.006  0.004  0.011  0.004  0.004  0.007  0.004
  0.005  0.006  0.007  0.006  0.008  0.01   0.012  0.011  0.005  0.004
  0.003  0.003  0.004  0.005  0.003  0.005  0.011  0.007  0.004  0.003
  0.003  0.006  0.004  0.002  0.004  0.004  0.004  0.007  0.002  0.004
  0.007  0.004  0.016  0.007  0.009  0.027  0.002  0.02   0.011  0.018
  0.009  0.008  0.012  0.004  0.004  0.008  0.009  0.03   0.03   0.005
  0.013  0.004  0.03   0.007  0.011  0.018  0.007  0.006  0.008  0.018
  0.022  0.037  0.008  0.007  0.006  0.03   0.007  0.008  0.005  0.013
  0.011  0.013  0.004  0.012  0.011  0.011  0.007  0.009  0.007  0.005]
[ 4.6    7.255]
[[b'5.1' b'3.5' b'1.4' b'0.2' b'Iris-setosa']
 [b'4.9' b'3.0' b'1.4' b'0.2' b'Iris-setosa']
 [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
 [b'4.6' b'3.1' b'1.5' b'0.2' b'Iris-setosa']
 [b'5.0' b'3.6' b'1.4' b'0.2' b'Iris-setosa']
 [b'5.4' b'3.9' b'1.7' b'0.4' b'Iris-setosa']
 [b'4.6' b'3.4' b'1.4' b'0.3' b'Iris-setosa']
 [b'5.0' b'3.4' b'1.5' b'0.2' b'Iris-setosa']
 [b'4.4' b'2.9' b'1.4' b'0.2' b'Iris-setosa']
 [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']]
[[b'5.1' b'3.5' b'1.4' b'0.2' b'Iris-setosa']
 [b'4.9' b'3.0' b'1.4' b'0.2' b'Iris-setosa']
 [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
 [b'4.6' b'3.1' b'1.5' b'0.2' b'Iris-setosa']
 [b'5.0' b'3.6' b'1.4' b'0.2' b'Iris-setosa']
 [b'5.4' b'3.9' b'1.7' b'0.4' b'Iris-setosa']
 [b'4.6' b'3.4' b'1.4' b'0.3' b'Iris-setosa']
 [b'5.0' b'3.4' b'1.5' b'0.2' b'Iris-setosa']
 [b'4.4' nan b'1.4' b'0.2' b'Iris-setosa']
 [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']]
Number of missing values: 
 5
Position of missing values: 
 (array([ 38,  80, 106, 113, 121]),)
[[ 4.8  3.4  1.6  0.2]
 [ 4.8  3.4  1.9  0.2]
 [ 4.7  3.2  1.6  0.2]
 [ 4.8  3.1  1.6  0.2]
 [ 4.9  2.4  3.3  1. ]
 [ 4.9  2.5  4.5  1.7]]
[[ 4.9  3.   1.4  0.2]
 [ 4.7  3.2  1.3  0.2]
 [ 4.6  3.1  1.5  0.2]
 [ 5.   3.6  1.4  0.2]
 [ 4.6  3.4  1.4  0.3]]
[[ 4.9  3.   1.4  0.2]
 [ 4.7  3.2  1.3  0.2]
 [ 4.6  3.1  1.5  0.2]
 [ 5.   3.6  1.4  0.2]
 [ 4.6  3.4  1.4  0.3]]
0.871754157305
0.871754157305
False
[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]
 [ 4.7  3.2  1.3  0.2]
 [ 4.6  3.1  1.5  0.2]]
(array([b'Iris-setosa', b'Iris-versicolor', b'Iris-virginica'], 
      dtype='|S15'), array([50, 50, 50]))
['small', 'small', 'small', 'small']
[[b'5.1' b'3.5' b'1.4' b'0.2' b'Iris-setosa' 38.13265162927291]
 [b'4.9' b'3.0' b'1.4' b'0.2' b'Iris-setosa' 35.200498485922445]
 [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa' 30.0723720777127]
 [b'4.6' b'3.1' b'1.5' b'0.2' b'Iris-setosa' 33.238050274980004]]
(array([b'Iris-setosa', b'Iris-versicolor', b'Iris-virginica'], dtype=object), array([77, 37, 36]))
1.7
[[b'4.3' b'3.0' b'1.1' b'0.1' b'Iris-setosa']
 [b'4.4' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
 [b'4.4' b'3.0' b'1.3' b'0.2' b'Iris-setosa']
 [b'4.4' b'2.9' b'1.4' b'0.2' b'Iris-setosa']
 [b'4.5' b'2.3' b'1.3' b'0.3' b'Iris-setosa']
 [b'4.6' b'3.6' b'1.0' b'0.2' b'Iris-setosa']
 [b'4.6' b'3.1' b'1.5' b'0.2' b'Iris-setosa']
 [b'4.6' b'3.4' b'1.4' b'0.3' b'Iris-setosa']
 [b'4.6' b'3.2' b'1.4' b'0.2' b'Iris-setosa']
 [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
 [b'4.7' b'3.2' b'1.6' b'0.2' b'Iris-setosa']
 [b'4.8' b'3.0' b'1.4' b'0.1' b'Iris-setosa']
 [b'4.8' b'3.0' b'1.4' b'0.3' b'Iris-setosa']
 [b'4.8' b'3.4' b'1.9' b'0.2' b'Iris-setosa']
 [b'4.8' b'3.4' b'1.6' b'0.2' b'Iris-setosa']
 [b'4.8' b'3.1' b'1.6' b'0.2' b'Iris-setosa']
 [b'4.9' b'2.4' b'3.3' b'1.0' b'Iris-versicolor']
 [b'4.9' b'2.5' b'4.5' b'1.7' b'Iris-virginica']
 [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']
 [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']]
b'1.5'
[50]
[ 27.63  14.64  21.8   30.    10.    10.    30.    30.    10.    29.18  30.
  11.25  10.08  10.    11.77  30.    30.    10.    30.    14.43]
[ 27.63  14.64  21.8   30.    10.    10.    30.    30.    10.    29.18  30.
  11.25  10.08  10.    11.77  30.    30.    10.    30.    14.43]
[ 4 13  5  8 17 12 11 14 19  1  2  0  9  6 16 18  7  3 10 15]
[15 10  3  7 18]
[ 41.    41.47  42.39  44.67  48.95]
[ 41.    41.47  42.39  44.67  48.95]
[ 41.    41.47  42.39  44.67  48.95]
[ 48.95  44.67  42.39  41.47  41.  ]
[[ 9  9  4  8  8  1  5  3  6  3]
 [ 3  3  2  1  9  5  1 10  7  3]
 [ 5  2  6  4  5  5  4  8  2  2]
 [ 8  8  1  3 10 10  4  3  6  9]
 [ 2  1  8  7  3  1  9  3  6  2]
 [ 9  2  6  5  3  9  4  6  1 10]]
[ 1  2  3  4  5  6  7  8  9 10]
[[1, 0, 2, 1, 1, 1, 0, 2, 2, 0], [2, 1, 3, 0, 1, 0, 1, 0, 1, 1], [0, 3, 0, 2, 3, 1, 0, 1, 0, 0], [1, 0, 2, 1, 0, 1, 0, 2, 1, 2], [2, 2, 2, 0, 0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 2, 0, 0, 2, 1]]
[' ' 'a' 'b' 'c' 'd' 'e' 'h' 'i' 'j' 'l' 'm' 'n' 'o' 'r' 't' 'y']
[[1, 0, 1, 1, 0, 0, 0, 2, 0, 3, 0, 2, 1, 0, 1, 0], [0, 2, 0, 0, 2, 1, 0, 1, 0, 0, 1, 2, 1, 2, 0, 0], [0, 4, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1]]
array_of_arrays:  [array([0, 1, 2]) array([3, 4, 5, 6]) array([7, 8, 9])]
[0 1 2 3 4 5 6 7 8 9]
[0 1 2 3 4 5 6 7 8 9]
[2 3 2 2 2 1]
[[ 0.  1.  0.]
 [ 0.  0.  1.]
 [ 0.  1.  0.]
 [ 0.  1.  0.]
 [ 0.  1.  0.]
 [ 1.  0.  0.]]
[[0 1 0]
 [0 0 1]
 [0 1 0]
 [0 1 0]
 [0 1 0]
 [1 0 0]]
['Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
 'Iris-versicolor' 'Iris-virginica' 'Iris-virginica' 'Iris-virginica'
 'Iris-virginica' 'Iris-virginica' 'Iris-virginica']
['Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
 'Iris-versicolor' 'Iris-virginica' 'Iris-virginica' 'Iris-virginica'
 'Iris-virginica' 'Iris-virginica' 'Iris-virginica']
[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
Array:  [ 9  4 15  0 17 16 17  8  9  0]
[4 2 6 0 8 7 9 3 5 1]
[[ 9  4 15  0 17]
 [16 17  8  9  0]]
[[4 2 6 0 8]
 [7 9 3 5 1]]
[[9 9 4]
 [8 8 1]
 [5 3 6]
 [3 3 3]
 [2 1 9]]
[9 8 6 3 9]
[9 8 6 3 9]
[[9 9 4]
 [8 8 1]
 [5 3 6]
 [3 3 3]
 [2 1 9]]
[ 0.44  0.12  0.5   1.    0.11]
[0 0 3 0 2 4 2 2 2 2]
[False  True False  True False False  True  True  True  True]
[[b'Iris-setosa', 3.4180000000000001], [b'Iris-versicolor', 2.7700000000000005], [b'Iris-virginica', 2.9740000000000002]]
[[b'Iris-setosa', 3.4180000000000001], [b'Iris-versicolor', 2.7700000000000005], [b'Iris-virginica', 2.9740000000000002]]
[ 1.  2.  3.  5.  6.  7.]
6.7082039325
[2 5]
[[2 2 2]
 [2 2 2]
 [2 2 2]]
8
8
2018-02-25 22:10:10
2018-02-25 22:10:10
array:  [8 8 3 7 7 0 4 2 5 2]
[ 6.33  6.    5.67  4.67  3.67  2.    3.67  3.  ]
[ 6.33  6.    5.67  4.67  3.67  2.    3.67  3.  ]
[ 5  8 11 14 17 20 23 26 29 32]
['2018-02-01' '2018-02-03' '2018-02-05' '2018-02-07' '2018-02-09'
 '2018-02-11' '2018-02-13' '2018-02-15' '2018-02-17' '2018-02-19'
 '2018-02-21' '2018-02-23']
['2018-02-01' '2018-02-02' '2018-02-03' '2018-02-04' '2018-02-05'
 '2018-02-06' '2018-02-07' '2018-02-08' '2018-02-09' '2018-02-10'
 '2018-02-11' '2018-02-12' '2018-02-13' '2018-02-14' '2018-02-15'
 '2018-02-16' '2018-02-17' '2018-02-18' '2018-02-19' '2018-02-20'
 '2018-02-21' '2018-02-22' '2018-02-23']
['2018-02-01' '2018-02-02' '2018-02-03' '2018-02-04' '2018-02-05'
 '2018-02-06' '2018-02-07' '2018-02-08' '2018-02-09' '2018-02-10'
 '2018-02-11' '2018-02-12' '2018-02-13' '2018-02-14' '2018-02-15'
 '2018-02-16' '2018-02-17' '2018-02-18' '2018-02-19' '2018-02-20'
 '2018-02-21' '2018-02-22' '2018-02-23']
[[ 0  1  2  3]
 [ 2  3  4  5]
 [ 4  5  6  7]
 [ 6  7  8  9]
 [ 8  9 10 11]
 [10 11 12 13]]
[Finished in 2.9s]
"""
