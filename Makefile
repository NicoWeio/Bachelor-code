all: build/sklearn_nn__hist_log.pdf
# all: print

# print:
# 	@echo "This Makefile is not up to date. Run directly."

build_large/data.csv: a_data_selection.py a_data_selection_features.csv | build_large/
	python a_data_selection.py

build_large/eval.hdf5: c_corn.py build_large/data.csv
	python c_corn.py

build/sklearn_nn__hist_log.pdf: d_evaluate.py build_large/eval.hdf5
	python d_evaluate.py
