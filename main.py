import data_preprocess as dp
import sklearn.linear_model as lr
import sklearn.decomposition as dc
import sklearn.preprocessing as sp
import numpy as np


def get_average_score_of_lr_by_all_features(X, y, times=1000):
    """
    get the average R2 score of a linear regression model
    :param X: feature matrix
    :param y: labels
    :param times: iteration times
    :return: average R2 score of the linear regression model
    """
    X_normalized = sp.normalize(X, axis=0)
    sum_of_test_score = 0
    number_of_samples = len(X)
    set_separation_index = int(number_of_samples * 0.7)  # separate data into to data set

    for i in range(times):
        linear_regression_model = lr.LinearRegression(normalize=True)
        X_and_y = np.hstack((X_normalized, y))
        np.random.shuffle(X_and_y)

        X_training = X_and_y[0:set_separation_index, 0:-1]
        y_training = X_and_y[0:set_separation_index, -1:number_of_samples]
        X_test = X_and_y[set_separation_index:-1, 0:-1]
        y_test = X_and_y[set_separation_index:-1, -1:number_of_samples]

        # print("*******LR********")
        linear_regression_model.fit(X=X_training, y=y_training)
        # print(linear_regression_model.score(X=X_training, y=y_training))
        # print(linear_regression_model.score(X_test, y_test))
        # print(linear_regression_model.coef_)

        sum_of_test_score += linear_regression_model.score(X_test, y_test)

    return sum_of_test_score / times


def get_average_score_of_lr_by_one_column(X, y, col_index, times=1000):
    """
    Build LR model based on single feature.
    :param X: Feature matrix
    :param y: labels
    :param col_index: the column index of the feature selected
    :return: None
    """
    X_normalized = sp.normalize(X, axis=0)
    sum_of_test_score = 0

    number_of_samples = len(X)
    set_separation_index = int(number_of_samples * 0.7)  # separate data into to data set
    for i in range(times):
        linear_regression_model = lr.LinearRegression(normalize=True)
        X_and_y = np.hstack((X_normalized, y))
        np.random.shuffle(X_and_y)

        X_training = X_and_y[0:set_separation_index, col_index:col_index + 1]
        y_training = X_and_y[0:set_separation_index, -1:number_of_samples]
        X_test = X_and_y[set_separation_index:-1, col_index:col_index + 1]
        y_test = X_and_y[set_separation_index:-1, -1:number_of_samples]

        # print("*******LR********")
        linear_regression_model.fit(X=X_training, y=y_training)
        # print(linear_regression_model.score(X=X_training, y=y_training))
        # print(linear_regression_model.score(X_test, y_test))
        # print(linear_regression_model.coef_)

        sum_of_test_score += linear_regression_model.score(X_test, y_test)

    return sum_of_test_score / 1000


def get_average_score_of_lr_by_pca_ed_features(X, y, times=1000):
    """
    Essentially the same as get_average_score_of_lr_by_all_features(). Rename it for
    clarity
    :param X: feature matrix
    :param y: labels
    :param times: iteration times
    :return: average R2 score of the linear regression model
    """
    X_normalized = sp.normalize(X, axis=0)
    sum_of_test_score = 0
    number_of_samples = len(X)
    set_separation_index = int(number_of_samples * 0.7)  # separate data into to data set
    for i in range(times):
        linear_regression_model = lr.LinearRegression(normalize=True)
        X_and_y = np.hstack((X_normalized, y))
        np.random.shuffle(X_and_y)

        X_training = X_and_y[0:set_separation_index, 0:-1]
        y_training = X_and_y[0:set_separation_index, -1:number_of_samples]
        X_test = X_and_y[set_separation_index:-1, 0:-1]
        y_test = X_and_y[set_separation_index:-1, -1:number_of_samples]

        # print("*******LR********")
        linear_regression_model.fit(X=X_training, y=y_training)
        # print(linear_regression_model.score(X=X_training, y=y_training))
        # print(linear_regression_model.score(X_test, y_test))
        # print(linear_regression_model.coef_)

        sum_of_test_score += linear_regression_model.score(X_test, y_test)

    return sum_of_test_score / times


def try_with_pca(X, y):
    """
    Training linear regression model after feature matrix are processed by PCA
    :param X: feature matrix
    :return: None
    """
    X_normalized = sp.normalize(X)
    number_of_features = len(X[0])
    for i in range(number_of_features - 1):
        print("******* " + str(i + 1) + " PCA *******")
        pca_model = dc.PCA(n_components=i + 1)
        X_temp = pca_model.fit_transform(X_normalized)
        print(pca_model.singular_values_)
        print("------  After PCA  ------")
        print(get_average_score_of_lr_by_pca_ed_features(X_temp, y))
        print()


def main():
    X, y, keys, feature_names = dp.load_and_preprocess_data()
    scr_hdi = 0
    k = 0
    print(get_average_score_of_lr_by_all_features(X, y))
    try_with_pca(X, y)
    for i in range(5):
        temp = get_average_score_of_lr_by_one_column(X, y, i)
        k += temp
        print(temp)

    print("----separate line----")
    for i in range(10):
        scr_hdi += get_average_score_of_lr_by_one_column(X, y, 0)

    print(scr_hdi / 100)


if __name__ == '__main__':
    main()
