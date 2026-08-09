# SpheresDB

A simple in-process vector database with optionally builtin Qwen 3 embedding 0.6B

mkdir build

cd build

cmake -DCMAKE_BUILD_TYPE=Release -G Ninja -S .. -B .

cmake --build . --target example
