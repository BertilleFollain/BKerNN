from Methods.BKRR import BKRR
from Methods.BKerNN import BKerNN
from Methods.ReLUNN import ReLUNN
from sklearn.utils.estimator_checks import check_estimator

for estimator in [BKRR(), ReLUNN(), BKerNN(reg_type='Basic'), BKerNN(reg_type='Variable'), BKerNN(reg_type='Feature'),
                  BKerNN(reg_type='Concave_Variable'), BKerNN(reg_type='Concave_Feature')]:
    for est, check in check_estimator(estimator, generate_only=True):
        print(str(check))
        try:
            check(est)
        except AssertionError as e:
            print('Failed: ', check, e)
    print(str(estimator) + ' Passed the Scikit-learn Estimator tests')
