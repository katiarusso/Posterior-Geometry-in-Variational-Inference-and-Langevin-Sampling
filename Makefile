.PHONY: setup reproduce test

setup:
	python -m pip install -e ".[dev]"

reproduce:
	python experiments/experiment_1.py
	python experiments/experiment_2.py

test:
	python -m unittest discover -s tests -v
