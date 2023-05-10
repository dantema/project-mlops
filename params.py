from hyperopt import fmin, hp, tpe, Trials, space_eval

#define search space for hyperparameters
search_space = {'learning_rate': hp.choice('learning_rate', [0.05]),
                'iterations': hp.choice('iterations', [100]),
                'l2_leaf_reg': hp.choice('l2_leaf_reg', [0, 1, 2 ]),
                'depth': hp.choice('depth', [8]),
                'bootstrap_type' : hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'random_seed' : 1, 
                'task_type': "CPU",
                'allow_writing_files' : False,
                'silent' : True
                }

#define arguments for fit model
fit_params = {'early_stopping_rounds' : 200}