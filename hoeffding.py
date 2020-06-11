import numpy as np

def hoeffding_inequality(nu, mu, N, eps):
    left_ine = 1 if (abs(nu - mu) > eps) else 0
    right_ine = 2 * np.exp(-2 * eps**2 * N)
    return (left_ine <= right_ine)

def hoeffding_check(nu, mu, N):
    eps_range = np.arange(0.01, 0.51, 0.01)
    for eps in eps_range:
        if (hoeffding_inequality(nu, mu, N, eps) == False):
            return False
    return True

def run_stimulation(n_test, n_coin, n_times):
    nus = np.random.binomial(n_times, 0.5, (n_test, n_coin)) / n_times
    v_1 = np.mean(nus[:, 0])
    pos_rand = np.random.choice(n_coin, size=n_test)
    v_rand = np.mean(nus[np.arange(n_test), pos_rand])
    v_min = np.mean(np.min(nus, axis=1))
    print("v_1 = ", v_1)
    print("v_rand = ", v_rand)
    print("v_)min = ", v_min)
    print("Checking Hoeffding inequality for v_1 : ", hoeffding_check(v_1, 0.5, 10))
    print("Checking Hoeffding inequality for v_rand : ", hoeffding_check(v_rand, 0.5, 10))
    print("Checking Hoeffding inequality for v_min : ", hoeffding_check(v_min, 0.5, 10))

if __name__ == '__main__':
    run_stimulation(n_test=100000, n_coin=1000, n_times=10)
