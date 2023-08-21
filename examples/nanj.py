import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import pandas as pd
from qlib.utils import exists_qlib_data
import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient


if __name__ == "__main__":
    
    
    # use default data
    provider_uri = "~/.qlib/qlib_data/gta"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    ###################################
    # train model
    ###################################

    market = "csi300"
    benchmark = "SH000300"


    data_handler_config = {
        "start_time": "2004-01-01",
        "end_time": "2023-08-01",
        "fit_start_time": "2004-01-01",
        "fit_end_time": "2017-12-31",
        "instruments": market,
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2004-01-01", "2017-12-31"),
                    "valid": ("2018-01-01", "2019-12-31"),
                    "test": ("2020-01-01", "2022-12-31"),
                },
            },
        },
    }

    # model initiaiton    
    model = init_instance_by_config(task["model"]) 
    dataset = init_instance_by_config(task["dataset"])
    
    
    ###################################
    # prediction, backtest & analysis
    ###################################

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2023-08-01",
            "account": 100000000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # # NOTE: This line is optional
    # # It demonstrates that the dataset can be used standalone.
    # example_df = dataset.prepare("train")
    # print(example_df.head())

    # start exp


    with R.start(experiment_name="zhanyuan",uri='http://localhost:5000') as run:
        # mlflow.autolog(log_models=False) 
        mlflow.lightgbm.autolog(log_models=False)

        R.log_params(**flatten_dict(task))
        model.fit(dataset)

        mlflow.sklearn.log_model(model,'LGBModel')
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()



