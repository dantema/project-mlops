from hyperopt import hp, tpe

split_params = {'test_size': 0.2,
                'random_state': 42}

#define search space for hyperparameters
search_space = {'learning_rate': hp.choice('learning_rate', [0.05, 0.1 , 0.2]),
                'iterations': hp.choice('iterations', [1000, 2000]),
                'l2_leaf_reg': hp.choice('l2_leaf_reg', [0, 1, 2 ]),
                'depth': hp.choice('depth', [6, 8]),
                'bootstrap_type' : hp.choice('bootstrap_type', 
                                             ['Bayesian', 'Bernoulli']),
                'random_seed' : 1, 
                'task_type': "CPU",
                'allow_writing_files' : False,
                'silent' : True
                }

#define arguments for fit model
fit_params = {'early_stopping_rounds' : 200}

fmin_args = {'space': search_space,
            'algo': tpe.suggest,
            'max_evals' : 20}

target_name = "price"

cat_features = ["model", "transmission", "fuelType"]