import time
from random import randint
from copy import deepcopy

import matplotlib.pyplot as plt

from hpbandster.core.nameserver import NameServer
from hpbandster.optimizers import BOHB
import hpbandster.visualization as hpvis

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
)


class BOHBTuner:
    def __init__(self, worker_klass, custom_config=None):
        self.worker_klass = worker_klass
        self.custom_config = custom_config if custom_config is not None else {}

    def get_configspace(self):
        cs = ConfigurationSpace()

        grid = self.worker_klass.get_grid(self.custom_config)
        param_list = []
        for key, value in grid.items():
            param = None
            value = deepcopy(value)
            # If it's a list, it's a discrete params
            if isinstance(value, list):
                param = CategoricalHyperparameter(key, value)
            elif isinstance(value, dict):
                if not "type" in value:
                    raise ValueError("Dict params for grid should have type")
                type_str = value.pop("type")
                if type_str == "int":
                    builder = UniformIntegerHyperparameter
                else:
                    builder = UniformFloatHyperparameter
                param = builder(key, **value)
            else: 
                raise ValueError(f"{key} is of unhandled type {type(value)}")
            param_list.append(param)
        cs.add_hyperparameters(param_list)

        return cs

    def fit(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        n_iter,
        min_budget,
        max_buget,
    ):

        run_id = str(int(time.time()))
        host = "127.0.0.1"
        port = randint(3000, 4000)
        # Step 1: Start a nameserver
        NS = NameServer(run_id=run_id, host=host, port=port)
        NS.start()

        # Step 2: Start a worker
        w = self.worker_klass(
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            run_id=run_id,
            nameserver=host,
            nameserver_port=port,
        )
        w.run(background=True)

        # Step 3: Run an optimizer
        bohb = BOHB(
            configspace=self.get_configspace(),
            run_id=run_id,
            nameserver=host,
            nameserver_port=port,
            min_budget=min_budget,
            max_budget=max_buget,
        )
        self.results = bohb.run(n_iterations=n_iter,)
        self.best_id = self.results.get_incumbent_id()
        # Step 4: Shutdown
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()

        # Step 5: Analysis
        id2config = self.results.get_id2config_mapping()

        inc_runs = self.results.get_runs_by_id(self.best_id)
        inc_run = inc_runs[-1]

        return {
            "best_params": id2config[self.best_id]["config"],
            "valid_score": inc_run.info["valid_score"],
            "test_score": inc_run.info["test_score"],
        }

    def describe_results(self):

        # get all executed runs
        all_runs = self.results.get_all_runs()

        # get the 'dict' that translates config ids to the actual configurations
        id2conf = self.results.get_id2config_mapping()

        # let's grab the run on the highest budget
        inc_runs = self.results.get_runs_by_id(self.best_id)
        inc_run = inc_runs[-1]

        # We have access to all information: the config, the loss observed during
        # optimization, and all the additional information
        inc_config = id2conf[self.best_id]["config"]

        print("Best found configuration:")
        print(inc_config)
        print(
            "It achieved accuracies of %f (validation) and %f (test)."
            % (inc_run.info["valid_score"], inc_run.info["test_score"])
        )

        print()
        print("A total of %i unique configurations were sampled." % len(id2conf.keys()))
        print("A total of %i runs where executed." % len(self.results.get_all_runs()))
        print()

        # Let's plot the observed losses grouped by budget,
        hpvis.losses_over_time(all_runs)

        # and the number of finished runs.
        hpvis.finished_runs_over_time(all_runs)

        # This one visualizes the spearman rank correlation coefficients of the losses
        # between different budgets.
        hpvis.correlation_across_budgets(self.results)

        # For model based optimizers, one might wonder how much the model actually helped.
        # The next plot compares the performance of configs picked by the model vs. random ones
        hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

        plt.show()
