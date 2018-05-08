
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