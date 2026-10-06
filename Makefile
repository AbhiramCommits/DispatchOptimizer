setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

data:
	python -m scripts.fetch_data

eda:
	python -m scripts.run_eda

eta:
	python -m scripts.train_eta

match:
	python -m scripts.run_matching

pareto:
	python -m scripts.run_pareto

bench:
	python -m scripts.run_benchmark
	python -m scripts.run_eta_sensitivity

test:
	pytest -q --maxfail=1

all: setup data eda eta match pareto bench test
