build-book:
	rm -rf docs
	rm -rf book/_build
	# list folders with notebooks here. Notebooks must be present in _toc.yml.
	cp -r spatial_unit book/spatial_unit
	jupyter-book build book
	cp -r book/_build/html docs
	rm -rf book/spatial_unit
	touch docs/.nojekyll
