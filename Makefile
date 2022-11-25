.PHONY: build

network:
	xenosite-fragment --max-size=7 bioactivation_dataset.csv reactive_ring_network.pkl.gz
