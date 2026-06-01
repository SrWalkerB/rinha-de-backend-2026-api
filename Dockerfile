# syntax=docker/dockerfile:1

# nlist baked into the prebuilt index. Bigger = smaller cells = faster query
# (lower p99), slower build. Build is offline (full CPU) so we can afford it.
ARG KNN_NLIST=4096
ARG KNN_KMEANS_ITERS=8

# ---- build stage ----
FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod ./
COPY main.go ./
COPY internal/ ./internal/
COPY cmd/ ./cmd/
COPY resources/normalization.json resources/mcc_risk.json resources/model.json ./resources/
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-s -w" -o /out/rinha-fraud . && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -o /out/buildindex ./cmd/buildindex

# Build the IVF index OFFLINE (no CPU cap here): k-means over the 3M vectors with
# a large nlist, serialized to index.bin. The big .gz is needed only at this
# step — it does NOT ship in the runtime image.
COPY resources/references.json.gz ./resources/references.json.gz
ARG KNN_NLIST
ARG KNN_KMEANS_ITERS
RUN REFERENCES_PATH=./resources/references.json.gz \
    INDEX_OUT=/out/index.bin \
    KNN_NLIST=${KNN_NLIST} \
    KNN_KMEANS_ITERS=${KNN_KMEANS_ITERS} \
    /out/buildindex

# ---- runtime stage ----
FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=build /out/rinha-fraud /app/rinha-fraud
# Prebuilt index baked into the image (replaces the raw dataset). Startup just
# reads it — no k-means under the CPU cap.
COPY --from=build /out/index.bin /app/index.bin
ENV INDEX_PATH=/app/index.bin \
    ADDR=:8080
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/app/rinha-fraud"]
