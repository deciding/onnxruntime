#!/bin/bash
#./build.sh --config Debug --build_shared_lib --parallel --skip_onnx_tests --skip_tests
#./build.sh --config RelWithDebInfo --build_shared_lib --parallel --skip_onnx_tests --skip_tests
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --skip_onnx_tests --skip_tests --build_wheel
