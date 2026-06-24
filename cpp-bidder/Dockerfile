# Reproducible build environment for the C++ bid server. Getting gRPC,
# protobuf, ONNX Runtime, prometheus-cpp and yaml-cpp to coexist is the most
# fragile part of Phase 2, so we pin it all in a container.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Toolchain plus gRPC, protobuf, yaml-cpp and nlohmann/json from apt.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    pkg-config \
    libgrpc++-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    libprotobuf-dev \
    libyaml-cpp-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime is not in apt: download the prebuilt CPU build.
ARG ORT_VERSION=1.18.0
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
    && tar -xzf onnxruntime-linux-x64-${ORT_VERSION}.tgz -C /opt \
    && mv /opt/onnxruntime-linux-x64-${ORT_VERSION} /opt/onnxruntime \
    && rm onnxruntime-linux-x64-${ORT_VERSION}.tgz

# prometheus-cpp is not in apt either. Build the pull (scrape) library from
# source; it bundles civetweb as a submodule for the /metrics HTTP endpoint.
ARG PROM_VERSION=v1.2.4
RUN git clone --depth 1 --branch ${PROM_VERSION} https://github.com/jupp0r/prometheus-cpp.git /tmp/prometheus-cpp \
    && cd /tmp/prometheus-cpp \
    && git submodule init && git submodule update --depth 1 \
    && cmake -S . -B build -DBUILD_SHARED_LIBS=ON -DENABLE_TESTING=OFF -DENABLE_PUSH=OFF \
    && cmake --build build --parallel 4 \
    && cmake --install build \
    && ldconfig \
    && rm -rf /tmp/prometheus-cpp

WORKDIR /app
COPY . .

# limit parallelism: the gRPC async + generated proto templates are memory-heavy
# to compile, and a full -j can spike RAM enough to destabilize the Docker VM.
RUN cmake -S . -B build && cmake --build build -j2

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/lib

EXPOSE 50051 9100
CMD ["./build/bid_server", "config/config.yaml"]
