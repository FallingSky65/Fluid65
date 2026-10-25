mkdir -p build-web && cd build-web
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ..
