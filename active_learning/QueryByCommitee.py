from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
from Oracle import get_points


class ContinueI(Exception):
    pass


class QBC:
    @staticmethod
    def get_labels(x):
        assert x.shape[1] == 10
        return get_points(x)

    def predict(self, X):
        return np.mean([self.estimators[i].predict(X) for i in range(self.n_estimators)], 0)

    # Original query by commitee with std as sorting feature
    def get_next_points_idxs_QBC(self, X, n=1):
        results = np.hstack([self.estimators[i].predict(X).reshape(-1, 1) for i in range(self.n_estimators)])
        stds = np.std(results, 1)
        return np.argsort(stds)[-n:]

    # Idea of working with std's and absolute values altogether
    def get_next_points_idxs_QBCreg(self, X, n=1):
        results = np.hstack([self.estimators[i].predict(X).reshape(-1, 1) for i in range(self.n_estimators)])
        stds = np.std(results, 1) + np.max(np.abs(results), axis=1) ** 2
        return np.argsort(stds)[-n:]

    def get_along_grad(self, pt, grad, multiplier, n):
        X_list = []
        grad = grad * multiplier
        for i in range(n):
            pt = pt + grad
            if pt.max() > 10 or pt.min() < 0:
                break
            X_list.append(pt)
        if len(X_list) == 0:
            return None, None
        X = np.vstack(X_list)
        y = self.get_labels(X)
        y = y[~np.isnan(y)]
        X = X[~np.isnan(y), :]
        return X, y


    def get_next_points_gradient_walk(self, X, y, absmin=150):
        continue_i = ContinueI()
        # Idea: if abs(value) for the point is greater than absmin => sample points along gradient
        big_value_points = X[np.abs(y) > absmin]
        big_value_labels = y[np.abs(y) > absmin]
        if big_value_points.shape[0] == 0:
            return None, None
        eps = 1e-6
        result_points = []
        result_labels = []
        bad_counter = 0  # Statistics. Points with nan's should be counted as calculated

        # Trying to get gradient of the function
        for i in range(big_value_points.shape[0]):
            try:
                pt = big_value_points[i, :].reshape(1, -1)
                grad = np.zeros([1, X.shape[1]])
                for j in range(X.shape[1]):
                    grad_step = np.zeros([1, X.shape[1]])
                    grad_step[0, j] = eps
                    grad_point = pt + grad_step
                    grad_label = self.get_labels(grad_point)
                    if not np.isnan(grad_label):
                        result_points.append(grad_point)
                        result_labels.append(grad_label)
                        grad[0, j] = (grad_label[-1] - big_value_labels[i])/eps
                    else:
                        bad_counter += j
                        raise continue_i

                # Normalising gradient
                gradnorm = (np.sqrt((grad**2).sum()))
                if gradnorm < 1:
                    raise continue_i
                grad = grad/gradnorm

                # Getting points along gradient
                X_plus, y_plus = self.get_along_grad(pt, grad, 0.0001, 25)
                X_minus, y_minus = self.get_along_grad(pt, grad, -0.0001, 25)
                curr_X = []
                curr_y = []
                if X_plus is not None:
                    curr_X.append(X_plus)
                    curr_y.append(y_plus)
                if X_minus is not None:
                    curr_X.append(X_minus)
                    curr_y.append(y_minus)

                if len(curr_X) != 0:
                    curr_X = np.vstack(curr_X)
                    curr_y = np.hstack(curr_y)
                    result_points.append(curr_X)
                    result_labels.append(curr_y)

                ''' 
                Next idea: if gradient has more than one nonzero direction, we can sample some points which
                are nearly orthogonal in the plane of the two most valuable coordinates 
                '''
                if np.max(np.abs(grad))**2 < 0.8:
                    max_idxs = np.argsort(np.abs(grad))[::-1][:2]
                    a_0 = grad[0, max_idxs[0, 0]]
                    a_1 = grad[0, max_idxs[0, 1]]
                    orthgrad = grad.copy()
                    orthgrad[0, max_idxs[0, 0]] = a_1
                    orthgrad[0, max_idxs[0, 1]] = -a_0
                    grad = orthgrad.copy()

                    X_plus, y_plus = self.get_along_grad(pt, grad, 0.001, 12)
                    X_minus, y_minus = self.get_along_grad(pt, grad, -0.001, 12)
                    curr_X = []
                    curr_y = []
                    if X_plus is not None:
                        curr_X.append(X_plus)
                        curr_y.append(y_plus)
                    if X_minus is not None:
                        curr_X.append(X_minus)
                        curr_y.append(y_minus)

                    if len(curr_X) != 0:
                        curr_X = np.vstack(curr_X)
                        curr_y = np.hstack(curr_y)
                        result_points.append(curr_X)
                        result_labels.append(curr_y)
            except ContinueI:
                continue
        res_X, res_y = np.vstack(result_points), np.hstack(result_labels)
        return res_X, res_y

    # Sample points by max_std between estimators/max abs/random respectively to weights
    def get_next_points_idxs_QBC2reg(self, X, n=1, weights=(9, 1, 10)):
        if n % np.sum(weights) != 0:
            print("Last pts")
            return self.get_next_points_idxs_QBC(X, n)
        to_parse = np.array([n / np.sum(weights) * weight for weight in weights]).astype(int)
        results = np.hstack([self.estimators[i].predict(X).reshape(-1, 1) for i in range(self.n_estimators)])
        stds = np.std(results, 1)
        stds_idxs = np.argsort(stds)[-to_parse[0]:].tolist()
        abses_idxs = np.argsort(np.max(np.abs(results), axis=1))[::-1].tolist()
        counter_abses = 0
        cur_idx = 0
        while counter_abses < to_parse[1]:
            if abses_idxs[cur_idx] not in stds_idxs:
                stds_idxs.append(abses_idxs[cur_idx])
                counter_abses += 1
            cur_idx += 1
        counter_rand = 0
        while counter_rand < to_parse[2]:
            num = np.random.randint(0, X.shape[0])
            if num not in stds_idxs:
                stds_idxs.append(num)
                counter_rand += 1
        assert (len(stds_idxs) == n)
        return np.array(stds_idxs).astype(int)

    # Random sampling
    def get_next_points_idxs_random(self, X, n=1):
        return np.random.choice(np.arange(X.shape[0]), n)

    def __init__(self, base_estimator=DecisionTreeRegressor,  n_estimators=2, batch_size=5, random_state=42,
                 sampling="QBC", init_size=50, to_label=500, use_grad=True, use_boundaries=True, **estimator_args):
        sampling_func_dict = {"QBC": self.get_next_points_idxs_QBC,
                              "random": self.get_next_points_idxs_random,
                              "QBCreg": self.get_next_points_idxs_QBCreg,
                              "QBC2reg": self.get_next_points_idxs_QBC2reg}
        self.sampling_func = sampling_func_dict[sampling]
        self.n_estimators = n_estimators
        if isinstance(base_estimator, DecisionTreeRegressor.__class__):
            self.estimators = [base_estimator(**estimator_args) for i in range(n_estimators)]
        if isinstance(base_estimator, XGBRegressor.__class__):
            self.estimators = [base_estimator(**estimator_args) for i in range(n_estimators)]
        self.random_state = random_state
        self.batch_size = batch_size
        np.random.seed(random_state)
        self.init_size = init_size
        self.to_label = to_label
        self.use_grad = use_grad
        self.use_boundaries = use_boundaries

    # Fitting ensemble
    def __subfit(self, X, y):
        if self.n_estimators == 1:
            self.estimators[0].fit(X, y)
        else:
            perm = np.random.permutation(X.shape[0])
            X = X[perm, :]
            y = y[perm]
            fold_size = int(X.shape[0] / self.n_estimators)
            small_folds_idxs = []
            for i in range(self.n_estimators - 1):
                small_folds_idxs.append(np.arange(i * fold_size, (i + 1) * fold_size))
            small_folds_idxs.append(np.arange(self.n_estimators - 1 * fold_size, X.shape[0]))
            folds_idxs = []
            for i in range(self.n_estimators):
                folds_idxs.append(np.hstack(small_folds_idxs[:i] + small_folds_idxs[i + 1:]))

            for i in range(self.n_estimators):
                self.estimators[i].fit(X[folds_idxs[i], :], y[folds_idxs[i]])

    # Step of the algorythm
    def __fit_active_step(self, X_labeled, X_unlabeled, y_labeled, batch_size=5):
        new_point_idxs = self.sampling_func(X_unlabeled, n=batch_size)
        mask = np.zeros(X_unlabeled.shape[0], dtype=bool)
        mask[new_point_idxs] = True
        new_features = X_unlabeled[mask, :]
        new_labels = self.get_labels(new_features)
        new_X_labeled = np.vstack([X_labeled, new_features])
        new_y_labeled = np.hstack([y_labeled, new_labels])
        if self.use_grad:
            new_gradfeat, new_gradlab = self.get_next_points_gradient_walk(new_features, new_labels)
            if new_gradfeat is not None:
                new_X_labeled = np.vstack([new_X_labeled, new_gradfeat])
                new_y_labeled = np.hstack([new_y_labeled, new_gradlab])
        nanmask = np.isnan(new_y_labeled)
        new_X_labeled = new_X_labeled[~nanmask]
        new_y_labeled = new_y_labeled[~nanmask]
        self.__subfit(new_X_labeled, new_y_labeled)
        return new_X_labeled, X_unlabeled[~mask, :], new_y_labeled

    # Had idea of getting boundaries of used hypercube. No achievements
    def __get_boundaries(self, min=1e-1, max=10 - 1e-1,  margin=1e-6, step=0.95):
        size_line = int((max - min) // step) + 1
        size = size_line * 10
        X = np.zeros((size, 10))
        X.fill(margin)
        for i in range(10):
            X[i*size_line:(i+1)*size_line, i] = np.arange(min, max, step)
        return X

    def fit(self, X):
        '''
        Step 1: get some random points
        '''
        first_n = self.init_size
        first_point_idxs = np.random.choice(np.arange(X.shape[0]), first_n)
        points_mask = np.zeros(X.shape[0], dtype=bool)
        points_mask[first_point_idxs] = True
        curr_labeled_points = X[points_mask, :]
        curr_unlabeled_points = X[~points_mask, :]
        curr_labels = self.get_labels(curr_labeled_points)
        if self.use_boundaries:
            boundaries = self.__get_boundaries()
            boundaries_labels = self.get_labels(boundaries)
            curr_labeled_points = np.vstack([curr_labeled_points, boundaries])
            curr_labels = np.hstack([curr_labels, boundaries_labels])

        nanmask = np.isnan(curr_labels)
        curr_labels = curr_labels[~nanmask]
        curr_labeled_points = curr_labeled_points[~nanmask]

        '''
        Step 2: loop
        '''
        self.__subfit(curr_labeled_points, curr_labels)
        points_left = self.to_label - curr_labeled_points.shape[0]
        while points_left > 0:
            curr_labeled_points, curr_unlabeled_points, curr_labels = self.__fit_active_step(curr_labeled_points,
                                                                                             curr_unlabeled_points,
                                                                                             curr_labels,
                                                                                             min(points_left,
                                                                                                 self.batch_size))
            points_left = self.to_label - curr_labeled_points.shape[0]
            print("Points: {}\tPart error: {}".format(curr_labeled_points.shape[0],
                                                      np.sqrt(mean_squared_error(
                                                          self.predict(curr_labeled_points), curr_labels))))
        print("Total points: ", curr_labeled_points.shape[0])

    def generate_prediction(self, values, name="sample_submission.txt"):
        result = self.predict(values)
        with open(name, 'w') as file:
            file.write("Id,Expected\n")
            for i in range(1,result.shape[0] + 1):
                file.write("{},{}\n".format(i, result[i-1]))
