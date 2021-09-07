build-book:
	rm -rf docs
	rm -rf book/_build
	# list folders with notebooks here. Notebooks must be present in _toc.yml.
	cp -r spatial_unit book/spatial_unit
	cp -r clusters book/clusters
	cp -r measuring book/measuring
	cp -r data_product book/data_product
	jupyter-book build book
	cp -r book/_build/html docs
	rm -rf book/spatial_unit
	rm -rf book/clusters
	rm -rf book/measuring
	rm -rf book/data_product
	touch docs/.nojekyll
