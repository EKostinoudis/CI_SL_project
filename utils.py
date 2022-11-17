from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import metrics
from time import time
import os

def calculate_metrics(model, x, y):
    y_predicted = model.predict(x)
    acc = metrics.accuracy_score(y, y_predicted)
    prec = metrics.precision_score(y, y_predicted, average="macro")
    recall = metrics.recall_score(y, y_predicted, average="macro")
    f1 = metrics.f1_score(y, y_predicted, average="macro")
    return (acc, prec, recall, f1)

class TrainTestMetric():
    def calculate_metrics(self, x, y):
        y_predicted = self.predict(x)
        acc = metrics.accuracy_score(y, y_predicted)
        prec = metrics.precision_score(y, y_predicted, average="macro")
        recall = metrics.recall_score(y, y_predicted, average="macro")
        f1 = metrics.f1_score(y, y_predicted, average="macro")
        return (acc, prec, recall, f1)

    def calculate_train_test_metrics(self, x_train, y_train, x_test, y_test):
        self.calculate_test_metrics(x_test, y_test)
        self.calculate_train_metrics(x_train, y_train)

    def calculate_train_metrics(self, x_train, y_train):
        self.train_metrics = self.calculate_metrics(x_train, y_train)

    def calculate_test_metrics(self, x_test, y_test):
        self.test_metrics = self.calculate_metrics(x_test, y_test)


class SVM_wrapper(SVC, TrainTestMetric):
    def __init__(self, kernel, **kwargs):
        super().__init__(kernel=kernel, **kwargs)
        # super().__init__(kernel=kernel, decision_function_shape='ovr', **kwargs)
        # super().__init__(kernel=kernel, decision_function_shape='ovo', **kwargs)
        self.init_kwargs = kwargs
    
    def fit(self, *args, **kwargs):
        t0 = time()
        super().fit(*args, **kwargs)
        self.train_time = time() - t0

class OVO_wrapper(OneVsOneClassifier, TrainTestMetric):
    def __init__(self, estimator, **kwargs):
        super().__init__(estimator, **kwargs)
        self.init_kwargs = kwargs
    
    def fit(self, *args, **kwargs):
        t0 = time()
        super().fit(*args, **kwargs)
        self.train_time = time() - t0
    
class GS_wrapper(GridSearchCV, TrainTestMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gs_init_args = args
        self.gs_init_kwargs = kwargs
    
    def fit(self, *args, **kwargs):
        t0 = time()
        super().fit(*args, **kwargs)
        self.gs_time = time() - t0

def save_plots(fig, name, folder=None):
    path = "figures"
    if folder is not None:
        path = os.path.join(path, folder)

    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, f"{name}.png"))
