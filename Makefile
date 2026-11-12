.PHONY: install test notebook clean

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/01_scorecard_woe_logistic.py -o notebooks/01_scorecard_woe_logistic.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/01_scorecard_woe_logistic.ipynb

clean:
	rm -rf .pytest_cache __pycache__ src/*.egg-info
