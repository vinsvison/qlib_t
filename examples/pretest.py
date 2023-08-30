from prefect import flow, task
from typing import List

@task(log_prints=True)
def say_hello(name: str):
    print(f"Hello {name}!")


@flow
def hello_universe(names: List[str]):
    for name in names:
        say_hello(name)

        
hello_universe(names=["Marvin", "Ford", "Arthur"])