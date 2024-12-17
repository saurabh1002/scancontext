.PHONY: cpp

install:
	@pip install --verbose ./python/

uninstall:
	@pip -v uninstall scan-context

cpp:
	@cmake -Bbuild .
	@cmake --build build -j$(nproc --all)
