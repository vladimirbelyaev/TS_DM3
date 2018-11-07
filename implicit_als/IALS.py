import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import product
import pickle
from datetime import datetime

class IALS:
    '''
    Based on the paper http://yifanhu.net/PUB/cf.pdf
    '''
    @staticmethod
    def __standard_confidence(R, alpha, eps):
        return np.ones(R.shape) + alpha * R

    @staticmethod
    def __log_confidence(R, alpha, eps):
        return np.ones(R.shape) + alpha * np.log(np.ones(R.shape) + R / eps)

    def __init__(self, max_epoch=10, embedding_size=15, alpha=10, l2reg=0.1, log_confidence=True, eps=0.1, random_state=42, verbose=1,
                 show_real_metric=False):
        self.max_epoch = max_epoch
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.l2reg = l2reg
        self.random_state = random_state
        self.verbose = verbose
        self.show_real_metric = show_real_metric
        self.eps = eps
        self.log_confidence = log_confidence
        if self.log_confidence:
            self.confidence_func = self.__log_confidence
        else:
            self.confidence_func = self.__standard_confidence
        np.random.seed(self.random_state)

    def eval_metrics(self, epoch, result, R_train, R_test, X, Y, C, gamma, beta, mean, subtract_mean, gr_mask, verbose):
        str_out = "Epoch {}\ttrain\t".format(epoch)
        train_error = np.sqrt(((result * gr_mask - R_train) ** 2).sum() / gr_mask.sum())
        str_out += str(train_error)
        if R_test is not None:
            test_error = np.sqrt(((result * (R_test > 0) - R_test) ** 2).sum() / (R_test > 0).sum())
            str_out += "\ttest\t{}".format(test_error)
        else:
            test_error = None
        if self.show_real_metric:
            metric = 1.0/(R.shape[0] * R.shape[1]) * \
                                  (C * (R - mean * subtract_mean - beta - gamma) ** 2).sum() + self.l2reg * \
                                  ((X**2).sum() + (Y**2).sum() + (beta**2).sum() + (gamma**2).sum())
            str_out += "\tmetric\t{}".format(metric)
        if verbose > 0:
            print(str_out)
        return train_error, test_error

    def fit(self, R_train, R_test=None):
        '''
        :param X:
            row - user
            column - item
        :return:
        '''

        # Generate user/item vectors
        gr_mask = R_train > 0
        eq_mask = R_train == 0
        mean = R_train[gr_mask].mean()
        # We either add mean to unknown values or subtract it from known and add it at the end
        subtract_mean = False
        if subtract_mean:
            P = R - gr_mask * mean
        else:
            P = R_train + eq_mask * mean
        # Precompute C and (C - 1) matrices
        C = self.confidence_func(R_train, self.alpha, self.eps)
        Cm1 = C - 1

        laI = np.eye(self.embedding_size + 1, self.embedding_size + 1) * self.l2reg  # fixed

        # Initialize embeddings
        X = np.hstack([np.ones((R_train.shape[0], 1)), np.random.random_sample((R_train.shape[0], self.embedding_size))])
        Y = np.hstack([np.ones((R_train.shape[1], 1)), np.random.random_sample((R_train.shape[1], self.embedding_size))])

        # Initialize biases
        beta = np.zeros((X.shape[0], 1))  # user bias
        gamma = np.zeros((X.shape[0], 1))  # item bias

        for epoch in range(self.max_epoch):
            # User-step
            # Precomputing some matrices
            YtYplusLaI = np.matmul(Y.T, Y) + laI
            Pgamma = P - gamma
            Cp = C * Pgamma
            for i in range(X.shape[0]):
                to_inv = YtYplusLaI + np.matmul(Y.T * Cm1[i, :], Y)
                inv_mat = np.linalg.inv(to_inv)
                inv_mat_to_y = np.matmul(inv_mat, Y.T)
                X[i, :] = np.matmul(inv_mat_to_y, Cp[i, :].reshape(-1, 1)).ravel()

            beta = X[:, 0].copy().reshape(-1, 1)
            X[:, 0] = 1
            XtXplusLaI = np.matmul(X.T, X) + laI
            Pbeta = P - beta
            Cp = C * Pbeta
            for j in range(Y.shape[0]):
                to_inv = XtXplusLaI + np.matmul(X.T * Cm1[:, j], X)
                inv_mat = np.linalg.inv(to_inv)
                inv_mat_to_x = np.matmul(inv_mat, X.T)
                Y[j, :] = np.matmul(inv_mat_to_x, Cp[:, j].reshape(-1, 1)).ravel()
            gamma = Y[:, 0].copy().reshape(1, -1)
            Y[:, 0] = 1
            epoch_result = np.matmul(X[:, 1:], Y[:, 1:].T) + beta + gamma + mean * subtract_mean
            if self.verbose > 0:
                if epoch % self.verbose == 0:
                    self.eval_metrics(epoch, epoch_result, R_train, R_test, X, Y, C,
                                      gamma, beta, mean, subtract_mean, gr_mask, self.verbose)
        train_error, test_error = self.eval_metrics(epoch, epoch_result, R_train, R_test, X, Y, C,
                                      gamma, beta, mean, subtract_mean, gr_mask, self.verbose)
        return epoch_result, train_error, test_error


def train_test_split(R, test_ratio=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    data_size = R.shape[0] * R.shape[1]
    nonzero_items = np.arange(data_size)[R.ravel() > 0]
    nonzero_size = nonzero_items.shape[0]
    test_size = int(test_ratio * nonzero_size)
    np.random.shuffle(nonzero_items)
    train_items = nonzero_items[test_size:]
    test_items = nonzero_items[:test_size]
    train_indices = np.unravel_index(train_items, R.shape)
    test_indices = np.unravel_index(test_items, R.shape)
    train_mask = np.zeros(R.shape)
    for i in range(train_indices[0].shape[0]):
        train_mask[train_indices[0][i], train_indices[1][i]] = True
    test_mask = np.zeros(R.shape)
    for i in range(test_indices[0].shape[0]):
        test_mask[test_indices[0][i], test_indices[1][i]] = True
    return R * train_mask, R * test_mask


if __name__ == "__main__":
    R = np.load("data/train.npy")
    cross_validation = False
    submit = False
    submission_name = "data/submission.txt"
    start_time = datetime.now()
    if cross_validation:
        seeds = [13, 42, 777, 1234, 1000000]
        splits = [train_test_split(R, 0.1, seed) for seed in seeds]
        alphas = list(range(5, 19, 2))
        l2regs = [0.01]
        emb_sizes = list(range(5, 11))
        results = dict()
        for l2reg, emb_size, alpha in product(l2regs, emb_sizes, alphas):
            results.setdefault((l2reg, emb_size, alpha), dict())
            ials = IALS(max_epoch=100, embedding_size=emb_size, alpha=alpha, l2reg=l2reg, verbose=15)
            for ind, seed in enumerate(seeds):
                print("alpha = {}\temb_size = {}\tl2reg = {}\tseed = {}".format(alpha, emb_size, l2reg, seed))
                R_train, R_test = splits[ind]
                results[(l2reg, emb_size, alpha)][seed] = ials.fit(R_train, R_test)
        pickle.dump(results, "data/cv_pickle")
    else:
        R_train, R_test = train_test_split(R, 0.1, 1234)
        ials = IALS(max_epoch=100, embedding_size=4, alpha=7, verbose=30, l2reg=0.1, eps=0.08,
                    log_confidence=True, show_real_metric=False)
        if submit:
            res, _, _ = ials.fit(R)
            with open("data/test.txt", 'r') as file_in:
                with open("data/submission.txt", 'w') as file_out:
                    file_out.write("Id,Score\n")
                    for id, line in enumerate(file_in):
                        ui = [int(a) for a in line.strip().split('\t')]
                        assert len(ui) == 2
                        file_out.write("{},{}\n".format(id + 1, res[ui[0] - 1, ui[1] - 1]))
        else:
            res = ials.fit(R_train, R_test)
            print("Time elapsed: {}".format(datetime.now() - start_time))
