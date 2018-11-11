import numpy as np
#from xgboost import XGBRegressor
#from sklearn.metrics import mean_squared_error
#from sklearn.tree import DecisionTreeRegressor


'''
Based on the papers
Log-confidence:
http://yifanhu.net/PUB/cf.pdf
ALS-WR:
https://www.researchgate.net/publication/220788980_Large-Scale_Parallel_Collaborative_Filtering_for_the_Netflix_Prize
'''
class IALS:
    @staticmethod
    def __standard_confidence(R_train, R_test, alpha, eps):
        return np.ones(R_train.shape) + alpha * R_train

    @staticmethod
    def __log_confidence(R_train, R_test, alpha, eps):
        test_eq_mask = R_test == 0
        return (np.ones(R_train.shape) + alpha * np.log(np.ones(R_train.shape) + R_train / eps))# * test_eq_mask

    def __bias_normalisation(self, R, R_test):
        gr_mask = R > 0
        eq_mask = R == 0
        mean = R[gr_mask].mean()
        R_unbiased = R * gr_mask - gr_mask * mean
        user_bias = (R_unbiased.sum(1) / gr_mask.sum(1)).reshape(-1, 1)
        R_unuserbiased = R_unbiased * gr_mask - gr_mask * user_bias
        gr_mask_sum_0 = gr_mask.sum(0)
        gr_mask_sum_0[gr_mask_sum_0 == 0] = 1
        item_bias = (R_unuserbiased.sum(0) / gr_mask_sum_0).reshape(1, -1)
        if self.use_test_to_init:
            gr_mask_test = R_test > 0
            P = R + eq_mask * (~gr_mask_test) * (user_bias + item_bias + mean) * self.mean_decrease + eq_mask * \
                gr_mask_test * (user_bias + item_bias + mean)
        else:
            P = R + eq_mask * (user_bias + item_bias + mean) * self.mean_decrease
        return P, mean, gr_mask, eq_mask

    def __meanplus_normalisation(self, R, R_test):
        gr_mask = R > 0
        eq_mask = R == 0
        mean = R[gr_mask].mean()
        P = R + eq_mask * mean * self.mean_decrease
        return P, mean, gr_mask, eq_mask

    def __meanminus_normalisation(self, R, R_test):
        gr_mask = R > 0
        eq_mask = R == 0
        mean = R[gr_mask].mean()
        P = R - gr_mask * mean #* self.mean_decrease
        return P, mean, gr_mask, eq_mask

    def __init__(self, max_epoch=10, embedding_size=15, alpha=10, l2reg=0.1, log_confidence=True, eps=0.1,
                 normalisation="+bias", mean_decrease=0.85, use_test_to_init=True, random_state=42, verbose=1,
                 show_real_metric=False):
        '''
        :param max_epoch:       number of iterations
        :param embedding_size:  size of embedding vector for users and items
        :param alpha:           alpha in confidence level formula
        :param l2reg:           lambda in optimised metric
        :param log_confidence:  False: Cij = 1 + alpha * R
                                True: Cij = 1 + alpha * log(1 + R/eps)
        :param eps:             coefficient in log_confidence formula
        :param ии:   "+mean" for adding mean * mean_decrease to unlabeled pairs
                                "-mean" for subtracting mean * mean_decrease from unlabeled pairs
                                "+bias" for adding mean + user_bias + item_bias
        :param mean_decrease:   how much should we decrease init value for unlabeled pairs
        :param use_test_to_init if True, we use test matrix to calculate init values for unlabeled data
        :param random_state:    random state
        :param verbose:         verbosity level(0 - nothing printed, 1+ - prints loss each verbose epoch)
        :param show_real_metric: Shows real metric optimised by ALS(value must not increase)
        '''
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
        self.mean_decrease = mean_decrease
        self.use_test_to_init = use_test_to_init
        self.normalisation = normalisation
        normalisation_funcs = {"+bias": self.__bias_normalisation,
                               "+mean": self.__meanplus_normalisation,
                               "-mean": self.__meanminus_normalisation}
        self.normalisation_func = normalisation_funcs[self.normalisation]
        np.random.seed(self.random_state)

    def eval_metrics(self, epoch, result, R_train, R_test, X, Y, C, gamma, beta, mean, subtract_mean, gr_mask, verbose):
        str_out = "Epoch {}\ttrain\t".format(epoch)
        train_error = np.sqrt(((result * gr_mask - R_train) ** 2).sum() / gr_mask.sum())
        str_out += str(train_error)
        R = R_train.copy()
        if R_test is not None:
            test_error = np.sqrt(((result * (R_test > 0) - R_test) ** 2).sum() / (R_test > 0).sum())
            str_out += "\ttest\t{}".format(test_error)
            R += R_test
        else:
            test_error = None
        if self.show_real_metric:
            metric = 1.0/(R.shape[0] * R.shape[1]) * \
                     (C * (R - mean * subtract_mean * self.mean_decrease - beta - gamma) ** 2).sum() + \
                     self.l2reg * ((X**2).sum() + (Y**2).sum() + (beta**2).sum() + (gamma**2).sum())
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

        P, mean, gr_mask, eq_mask = self.normalisation_func(R_train, R_test)
        subtract_mean = self.normalisation == "-mean"
        # Precompute C and (C - 1) matrices
        C = self.confidence_func(R_train, R_test, self.alpha, self.eps)
        Cm1 = C - 1

        laI = np.eye(self.embedding_size + 1, self.embedding_size + 1) * self.l2reg  # fixed

        # Initialize embeddings
        X = np.hstack([np.ones((R_train.shape[0], 1)), np.random.random_sample((R_train.shape[0], self.embedding_size))])
        Y = np.hstack([np.ones((R_train.shape[1], 1)), np.random.random_sample((R_train.shape[1], self.embedding_size))])

        # Initialize biases
        beta = np.zeros((X.shape[0], 1))  # user bias
        gamma = np.zeros((X.shape[0], 1))  # item bias


        if self.use_test_to_init:
            n_user = (R_train + R_test > 0).sum(1)
            n_item = (R_train + R_test > 0).sum(0)
        else:
            n_user = (R_train > 0).sum(1)
            n_item = (R_train > 0).sum(0)

        for epoch in range(self.max_epoch):
            # User-step
            YtY = np.matmul(Y.T, Y)
            Pgamma = P - gamma
            '''Pgamma = P - gamma - beta  # По формуле'''
            Cp = C * Pgamma
            for i in range(X.shape[0]):  # user-loop
                to_inv = YtY + laI * (n_user[i]) + np.matmul(Y.T * Cm1[i, :], Y)
                '''
                to_inv = YtY + laI * (R_train.shape[1]) + np.matmul(Y.T * Cm1[i, :], Y)  # По формуле
                '''
                inv_mat = np.linalg.inv(to_inv)
                inv_mat_to_y = np.matmul(inv_mat, Y.T)
                X[i, :] = np.matmul(inv_mat_to_y, Cp[i, :].reshape(-1, 1)).ravel()
            # Item-step
            beta = X[:, 0].copy().reshape(-1, 1)
            X[:, 0] = 1
            XtX = np.matmul(X.T, X)
            Pbeta = P - beta
            '''Pbeta = P - beta - gamma # По формуле'''
            Cp = C * Pbeta
            for j in range(Y.shape[0]):  # item-loop
                to_inv = XtX + laI * (n_item[j]) + np.matmul(X.T * Cm1[:, j], X)
                '''
                to_inv = XtX + laI * (R_train.shape[0]) + np.matmul(X.T * Cm1[:, j], X)  # По формуле
                '''
                inv_mat = np.linalg.inv(to_inv)
                inv_mat_to_x = np.matmul(inv_mat, X.T)
                Y[j, :] = np.matmul(inv_mat_to_x, Cp[:, j].reshape(-1, 1)).ravel()
            gamma = Y[:, 0].copy().reshape(1, -1)
            Y[:, 0] = 1
            epoch_result = np.matmul(X[:, 1:], Y[:, 1:].T) + beta + gamma + mean * subtract_mean * self.mean_decrease
            epoch_result_fix = epoch_result.copy()
            epoch_result_fix[epoch_result_fix > 5] = 5
            epoch_result_fix[epoch_result_fix < 1] = 1
            if self.verbose > 0:
                if (epoch + 1) % self.verbose == 0:
                    self.eval_metrics(epoch, epoch_result_fix, R_train, R_test, X, Y, C,
                                      gamma, beta, mean, subtract_mean, gr_mask, self.verbose)
        train_error, test_error = self.eval_metrics(epoch, epoch_result_fix, R_train, R_test, X, Y, C,
                                      gamma, beta, mean, subtract_mean, gr_mask, self.verbose)

        return epoch_result_fix, train_error, test_error


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
