# syntax=docker/dockerfile:1

# ---- build stage ----
FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod ./
COPY main.go ./
COPY internal/ ./internal/
COPY cmd/ ./cmd/
COPY resources/normalization.json resources/mcc_risk.json ./resources/
# NB: GOAMD64=v3 foi testado e REVERTIDO — o compilador do Go não auto-vetoriza
# os loops de distância (só muda FMA/BMI), então v3 ficou ~16% MAIS LENTO no bench
# (ver microbench v1 27us vs v3 31us). SIMD real exigiria assembly à mão.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-s -w" -o /out/rinha-fraud . && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-s -w" -o /out/lb ./cmd/lb && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -o /out/buildindex ./cmd/buildindex

# Build the partitioned k-d index OFFLINE (no CPU cap here): bucket + per-bucket
# k-d tree over the 3M vectors, serialized to index.bin (~8s). The big .gz is
# needed only at this step — it does NOT ship in the runtime image.
COPY resources/references.json.gz ./resources/references.json.gz
RUN REFERENCES_PATH=./resources/references.json.gz \
    INDEX_OUT=/out/index.bin \
    /out/buildindex

# ---- runtime stage ----
FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=build /out/rinha-fraud /app/rinha-fraud
# Minimal L4 load balancer (same image, different entrypoint via compose). Lets
# the LB run on far less CPU than nginx, freeing it for the CPU-bound APIs.
COPY --from=build /out/lb /app/lb
# Prebuilt index baked into the image (replaces the raw dataset). Startup just
# reads it — no k-means under the CPU cap.
COPY --from=build /out/index.bin /app/index.bin
ENV INDEX_PATH=/app/index.bin \
    ADDR=:8080
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/app/rinha-fraud"]
