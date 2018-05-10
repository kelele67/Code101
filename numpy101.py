
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