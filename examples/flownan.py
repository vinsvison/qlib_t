from prefect import task, Flow
from tastnan import init, model_init, dataset_init, train, predict, signal_record,backtest_record
import mlflow
from prefect import variables


@Flow
def flownan():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("zhanyuan")

    with mlflow.start_run() as run:
        init()
        mlflow.lightgbm.autolog()

        model = model_init()

        dataset = dataset_init()

        train(model, dataset)

        pred, label = predict(model, dataset)

        signal_record(pred, label)
        
        backtest_record(pred, label, benchmark="SH000300")


if __name__ == "__main__":
    flownan()
