from Methods_with_monitoring.BKerNN import BKerNN
from Methods_with_monitoring.ExpKerNN import ExpKerNN
from Methods_with_monitoring.GaussianKerNN import GaussianKerNN
from sklearn.utils.estimator_checks import check_estimator

for estimator in [BKerNN(), ExpKerNN(), GaussianKerNN()]:
    for est, check in check_estimator(estimator, generate_only=True):
        print(str(check))
        try:
            check(est)
        except AssertionError as e:
            print('Failed: ', check, e)
    print(str(estimator) + ' Passed the Scikit-learn Estimator tests')
