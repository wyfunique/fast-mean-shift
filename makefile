CC = c++
lib_path = lib

pybind11_headers_flags = $(shell python3 -m pybind11 --includes)
pybind11_compilation_suffix = $(shell python3-config --extension-suffix)
eigen_headers_flags = -I$(shell pwd)/eigen-3.4.0
custom_headers_flags = -I$(shell pwd)

fast_mean_shift: 
	mkdir -p $(lib_path)
	$(eval target := $@)
	$(CC) -O3 -Wall -shared -std=c++11 -fopenmp -fPIC $(custom_headers_flags) $(pybind11_headers_flags) $(eigen_headers_flags) *.cpp -o $(lib_path)/$(target)$(pybind11_compilation_suffix)
	@echo "Succeeded building $(target)$(pybind11_compilation_suffix)."
	@echo "Please copy the file $(lib_path)/$(target)$(pybind11_compilation_suffix) into your python package installation directory."

clean:
	rm -r $(lib_path) 