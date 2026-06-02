# syntax=docker/dockerfile:1

# ---- build stage ----
FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod ./
COPY main.go ./
COPY internal/ ./internal/
COPY cmd/ ./cmd/
COPY resources/normalization.json resources/mcc_risk.json ./resources/
# GOAMD64=v3 (AVX2+FMA+BMI) no binário de runtime: o host de avaliação é um Mac
# Mini Late 2014 (Intel Haswell i5-4278U), a 1ª geração com v3 completo. Isso
# auto-vetoriza os loops de distância float64 (sobretudo o scan de centróides
# O(nlist) por query) -> corta CPU/query -> menos throttle CFS -> p99 menor.
# buildindex fica baseline (roda no host de build, não é crítico de performance).
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -trimpath -ldflags="-s -w" -o /out/rinha-fraud . && \
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
# Prebuilt index baked into the image (replaces the raw dataset). Startup just
# reads it — no k-means under the CPU cap.
COPY --from=build /out/index.bin /app/index.bin
ENV INDEX_PATH=/app/index.bin \
    ADDR=:8080
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/app/rinha-fraud"]
