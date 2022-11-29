.PHONY: build, coverage

network:
	xenosite-fragment --max-size=7 bioactivation_dataset.csv reactive_ring_network.pkl.gz

coverage:
	NUMBA_DISABLE_JIT=1 pytest --cov=xenosite && coverage lcov  
