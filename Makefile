clean:
	rm -rf COMBO.egg-info
	rm -rf .eggs
	rm -rf .pytest_cache

develop:
	python setup.py develop

install:
	python setup.py install

test:
	python setup.py test
	pylint --rcfile=.pylintrc tests combo