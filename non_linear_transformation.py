import numpy as np 

def generate_data_with_non_linear_transformation(N, w):
    bad_data = True

    while bad_data == True:
        X = np.random.uniform(-1, 1, (N, 2))
        X = np.hstack((np.ones((N, 1)), X)) 
        x1 = X[:, 1].reshape(N, 1)
        x2 = X[:, 2].reshape(N, 1)
        Y = np.sign(x1**2 + x2**2 - 0.6)
        if (0 not in Y): # Good data
            bad_data = False
    error_idx = np.random.choice(np.arange(N), size=int(0.1*N), replace=False)
    Y[error_idx, 0] *= -1
    return X, Y

def generate_data_with_quadratic(N, w):
    bad_data = True

    while bad_data == True:
        X = np.random.uniform(-1, 1, (N, 2))
        X = np.hstack((np.ones((N, 1)), X)) 
        x1 = X[:, 1].reshape(N, 1)
        x2 = X[:, 2].reshape(N, 1)
        X = np.hstack((X, x1**2))
        X = np.hstack((X, x2**2))
        Y = np.sign(np.dot(X, w))
        if (0 not in Y): # Good data
            bad_data = False
    print("before error : ", Y);
    error_idx = np.random.choice(np.arange(N), size=int(0.1*N), replace=False)
    print("error idx : ", error_idx)
    Y[error_idx, 0] *= -1
    print("after error : ", Y)
    return X, Y

w = np.array([-0.6, 0, 0, 1, 1]).reshape(-1, 1)
X, Y = generate_data_with_quadratic(20, w)

print(X)
