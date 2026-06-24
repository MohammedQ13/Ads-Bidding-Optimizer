# Build environment for the Go auction engine. It generates the Go gRPC stubs
# from the shared proto, then builds the engine.
FROM golang:1.22-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

# the protoc plugins that turn the .proto into Go code
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.34.1 \
    && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.4.0
ENV PATH="/go/bin:${PATH}"

# let the build fetch and pin dependencies as needed
ENV GOFLAGS=-mod=mod

WORKDIR /app
COPY . .

# generate gen/bidding/*.pb.go from the proto (go_package strips to gen/bidding)
RUN protoc --go_out=. --go_opt=module=rtbengine \
           --go-grpc_out=. --go-grpc_opt=module=rtbengine \
           -Iproto proto/bid_service.proto

# run the test suite as part of the build: a failing test fails the image
RUN go vet ./... && go test ./...

RUN go build -o engine ./cmd/engine

# the outcome log lives here, shared with the retraining worker in compose
RUN mkdir -p /data

EXPOSE 9200
CMD ["./engine", "config.yaml"]
