.PHONY: cpp

editable:
	SETUPTOOLS_ENABLE_FEATURES="legacy-editable" pip install --verbose --prefix=$(shell python3 -m site --user-base) --editable .

install:
	@pip install --verbose .

uninstall:
	@pip -v uninstall scan-context

cpp:
	@cmake -Bbuild scan_context/
	@cmake --build build -j$(nproc --all)
