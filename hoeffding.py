import numpy as np

def run_stimulation(n_test, n_coin, n_times):
    nus = np.random.binomial(n_times, 0.5, (n_test, n_coin)) / n_times
    v_1 = np.mean(nus[:, 0])
    pos_rand = np.random.choice(n_coin, size=n_test)
    v_rand = np.mean(nus[np.arange(n_test), pos_rand])
    print("v_1 : ", v_1)
    print("v_rand : ", v_rand)
    v_min = np.min(nus, axis=1)
    print("v_min : ", np.mean(v_min))
if __name__ == '__main__':
    run_stimulation(n_test=100000, n_coin=1000, n_times=10)
