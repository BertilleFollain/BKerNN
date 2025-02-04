import numpy as np
import openml
import pickle

suite = openml.study.get_suite(353)  
suite_2 = openml.study.get_suite(269)
task_ids = suite.tasks + suite_2.tasks + [4798, 360860] 
max_n_samples = 1000  # todo: adjust

for task_id in task_ids:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    print(f'Dataset name: {dataset.name}')
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=task.target_name
    )
    if dataset.name in ['geographical_origin_of_music', 'tecator', 'Yolanda', 'pah', 'semeion']:
        assert not any(categorical_indicator)  # all variables should be numerical
        X = np.asarray(X.to_numpy(), dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        print(np.shape(X))

        indices = np.random.choice(X.shape[0], min(max_n_samples, X.shape[0]), replace=False)
        X = X[indices]
        y = y[indices]
        res = {'X': X, 'y': y}

        # Save results
        pickle.dump(res, open('../../Experiments_results/Data_for_Experiment6/Experiment6data' + dataset.name + '.pkl', 'wb'))
