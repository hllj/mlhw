import numpy as np

def generate_target_w():
    """
    Generates target_w (the vector of parameters of f)
    from two random, uniformly distributed points in [-1, 1] x [-1, 1].

    Returns
    -------
    target_w : numpy array, shape (3, 1)
        The vector of parameters of f.
    """
    # Generate two points from a uniform distribution over [-1, 1]x[-1, 1]
    p1 = np.random.uniform(-1, 1, 2)
    p2 = np.random.uniform(-1, 1, 2)
    # Compute the target W from these two points
    target_w = np.array([p1[1]*p2[0] - p1[0]*p2[1], p2[1] - p1[1], p1[0] - p2[0]]).reshape((-1, 1))

    return target_w

def generate_data(N, target_w):
    """
    Generates a data set by generating random inputs and then using target_w to generate the
    corresponding outputs.

    Parameters
    ----------
    N : int
        The number of examples.
    target_w : numpy array, shape (3, 1)
        The vector of parameters of f.

    Returns
    -------
    X : numpy array, shape (N, 3)
        The matrix of input vectors (each row corresponds to an input vector); the first column of
        this matrix is all ones.
    Y : numpy array, shape (N, 1)
        The vector of outputs.
    """
    bad_data = True # `bad_data = True` means: data contain points on the target line
                    # (this rarely happens, but just to be careful)
                    # -> y's of these points = 0 (with np.sign);
                    #    we don't want this (y's of data must be -1 or 1)
                    # -> re-generate data until `bad_data = False`

    while bad_data == True:
        X = np.random.uniform(-1, 1, (N, 2))
        X = np.hstack((np.ones((N, 1)), X)) # Add 'ones' column
        Y = np.sign(np.dot(X, target_w))
        if (0 not in Y): # Good data
            bad_data = False

    return X, Y

def run_PLA(X, Y):
    """
    Runs PLA.

    Parameters
    ----------
    X : numpy array, shape (N, 3)
        The matrix of input vectors (each row corresponds to an input vector); the first column of
        this matrix is all ones.
    Y : numpy array, shape (N, 1)
        The vector of outputs.

    Returns
    -------
    w : numpy array, shape (3, 1)
        The vector of parameters of g.
    num_iterations : int
        The number of iterations PLA takes to converge.
    """
    w = np.zeros((X.shape[1], 1)) # Init w
    iteration = 0

    # TODO
    while True:
        Y_hat = np.sign(X.dot(w))
        x_pos, y_pos = np.where(Y_hat != Y)
        if len(x_pos) == 0:
            break
        random_x = np.random.choice(x_pos)
        w = w + Y[random_x, 0] * X[random_x, :].reshape(-1, 1)
        iteration += 1

    return w, iteration

def main(N): # You don't have to name this function "main", you can use other names
    """
    Parameters
    ----------
    N : int
        The number of training examples.
    """
    num_runs = 1000
    avg_num_iterations = 0.0 # The average number of iterations PLA takes to converge
    avg_test_err = 0.0 # The average test error of g - the final hypothesis picked by PLA

    for r in range(num_runs):
        # Generate target_w
        target_w = generate_target_w()

        # Generate training set
        X, Y = generate_data(N, target_w)

        # Run PLA to pick g
        w, num_iterations = run_PLA(X, Y)

        # Generate test set
        X_test, Y_test = generate_data(10000, target_w)

        # Test g
        test_err = np.mean(np.sign(np.dot(X_test, w)) != Y_test)

        # Update average values
        avg_num_iterations += (num_iterations * 1.0 / num_runs)
        avg_test_err += (test_err * 1.0 / num_runs)

    # Print results
    print('avg_num_iterations = %f' % (avg_num_iterations))
    print('avg_test_err = %f' % (avg_test_err))

if __name__ == '__main__':
    main(N=10)
    main(N=100)
