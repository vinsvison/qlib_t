from pprint import pprint

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
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return, calc_long_short_prec
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest, executor
from qlib.utils.time import Freq
from qlib.contrib.report import analysis_model, analysis_position
from qlib.contrib.report.analysis_position.report import (
    _report_figure,
    _calculate_report_data,
)
from qlib.contrib.report.analysis_position.score_ic import score_ic_graph, _get_score_ic
import plotly.io as pio
import os
import tempfile
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.data.handler import Alpha158


if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
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
    # dataset = init_instance_by_config(task["dataset"])
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": "csi300",
    }
    dataset = Alpha158(**data_handler_config)
    print(dataset.fetch(col_set="label", data_key=dataset.DK_L))
    dataset_conf = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": dataset,
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
    }
    dataset = init_instance_by_config(dataset_conf)
    df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    print(df)

    # start exp

    with R.start(experiment_name="zhanyuan", uri="http://localhost:5000") as run:
        # mlflow.autolog()
        mlflow.lightgbm.autolog()

        R.log_params(**flatten_dict(task))
        model.fit(dataset)

        # R.save_objects(trained_model=model)
        # rid = R.get_recorder().id

        # prediction

        pred = model.predict(dataset)
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")

        params = dict(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
        # del params["data_key"]
        label = dataset.prepare(**params)

        # Signal Analysis
        # label_col=0
        ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])

        for i, (date, value) in enumerate(ic.items()):
            # 使用log_metric记录日期和对应的值
            mlflow.log_metric("ic", value, step=i)
        for i, (date, value) in enumerate(ric.items()):
            # 使用log_metric记录日期和对应的值
            mlflow.log_metric("rank_ic", value, step=i)
        print(ic.head())

        FREQ = "day"
        STRATEGY_CONFIG = {
            "topk": 50,
            "n_drop": 5,
            # pred_score, pd.Series
            "signal": pred,
        }

        EXECUTOR_CONFIG = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        }

        backtest_config = {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": FREQ,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }

        # strategy object
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        # executor object
        executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
        # backtest
        portfolio_metric_dict, indicator_dict = backtest(
            executor=executor_obj, strategy=strategy_obj, **backtest_config
        )
        analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))

        # backtest info
        report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)
        report_df = report_normal.copy()
        fig_list = _report_figure(report_df)
        for i, fig in enumerate(fig_list):
            fig.update_layout(autosize=False, width=1500)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, "report_normal.png")
                pio.write_image(fig, temp_file)
                mlflow.log_artifact(temp_file)

        # analysis
        cum_return = _calculate_report_data(report_normal)

        metrics = [
            "cum_bench",
            "cum_return_wo_cost",
            "cum_return_w_cost",
            "return_wo_mdd",
            "return_w_cost_mdd",
            "cum_ex_return_wo_cost",
            "cum_ex_return_w_cost",
            "cum_ex_return_wo_cost_mdd",
            "cum_ex_return_w_cost_mdd",
            "turnover",
        ]

        from concurrent.futures import ThreadPoolExecutor

        def log_metric(metric):
            for i, (date, value) in enumerate(cum_return[metric].items()):
                mlflow.log_metric(metric, value, step=i)

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(log_metric, metrics)
