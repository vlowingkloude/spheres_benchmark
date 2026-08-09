# Spheres benchmark

Single-query vector-search harnesses for SpheresDB, FAISS, hnswlib,
DuckDB VSS, pgvector, and a brute-force C++ baseline. The benchmark uses
squared L2 ranking and writes one JSON metrics file per run.

## Build

Dependencies for SpheresDB, FAISS, hnswlib, DuckDB, and DuckDB VSS are pinned
and fetched by CMake. PostgreSQL's C client library is a system dependency.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

On Fedora, install `libpq-devel`; on Debian/Ubuntu, install `libpq-dev`.

## Data

Download and extract the TexMex corpora:

```sh
curl -LO ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
curl -LO ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xf sift.tar.gz
tar -xf gist.tar.gz
```

The official ground-truth files are valid for squared L2 because squaring a
non-negative L2 distance does not change nearest-neighbor ordering.

## Run

Common arguments are `--engine`, `--mode`, `--dataset`, `--path`,
`--distance squared_l2`, `--n_threads`, `--metrics`, and `--index`.
Dataset values are `sift1m` and `gist1m`. Example:

```sh
./build/spheres_benchmark \
  --engine spheres --mode exact \
  --dataset sift1m --path /data/sift --distance squared_l2 \
  --n_threads 4 --metrics results/sift_spheres.json \
  --index results/sift_spheres.db --spheres_n_ppage 0
```

For SpheresDB, `--mode exact` uses the current exact page bound, `--mode ann`
uses that bound with a page-probe budget, and `--mode legacy_prune` injects
the older `D - M > W` squared-distance shortcut for an explicitly approximate
comparison.

FAISS modes are `flat`, `hnsw`, `ivf`, and `lsh`. hnswlib and DuckDB use
`hnsw`. pgvector supports `flat` and `hnsw`; provide its password through
`PGPASSWORD` rather than the process argument list:

```sh
PGPASSWORD=... ./build/spheres_benchmark \
  --engine pgvector --mode flat --dataset sift1m --path /data/sift \
  --distance squared_l2 --n_threads 4 --metrics results/sift_pg_flat.json \
  --pg_host 192.168.64.7
```

DuckDB and pgvector HNSW runs validate their physical plans before timing and
fail if the optimizer does not select the HNSW index. pgvector's `<->`
operator returns L2 rather than squared L2, but produces the same ranking.

## Benchmark suite

`scripts/run_benchmarks.sh` configures and builds the explicit
`spheres_benchmark` target, then runs a selected set of engines on one dataset.
Each run receives a separate metrics file, index path, and log. A failed or
out-of-memory engine is recorded in a TSV summary without discarding later
runs.

```sh
scripts/run_benchmarks.sh \
  --dataset sift1m \
  --data /data/sift \
  --results results \
  --threads 4
```

The default set contains the brute-force baseline, SpheresDB exact search,
FAISS Flat/HNSW/IVFFlat, hnswlib, and DuckDB VSS. Select a smaller set with a
comma-separated list:

```sh
scripts/run_benchmarks.sh \
  --dataset gist1m \
  --data /data/gist \
  --engines spheres,faiss-flat,faiss-hnsw,hnswlib \
  --dry-run
```

pgvector is opt-in because it needs an external server:

```sh
PGPASSWORD=... scripts/run_benchmarks.sh \
  --dataset sift1m \
  --data /data/sift \
  --engines pgvector-flat,pgvector-hnsw \
  --pg-host 192.168.64.7
```

Use `--skip-build` to reuse an existing executable and `--fail-fast` to stop at
the first failed run. `--update-dashboard` regenerates the dashboard from the
specified results directory; that directory should contain the canonical
results for both datasets if the existing two-dataset dashboard must be
preserved.

## Measurement semantics

Queries are issued one at a time. This measures single-request latency, not
batch throughput. Local engines are in-process; pgvector latency includes the
client/server round trip. Build, save, load, and search phases are timed
separately where supported.

SpheresDB and the brute-force baseline also report `avg_squared_l2`, the mean
squared distance from returned neighbors to their queries, and
`avg_squared_l2_gt`, the corresponding mean for ground-truth neighbors. This
distinguishes approximate results with similar Recall@k but different distance
quality.

SpheresDB commit `1641bc0` implements an exact squared-L2 page bound without
evaluating square roots. With `d`, `m`, and `w` as the true query-to-anchor
distance, page radius, and current worst distance, the exact test is
`d - m > w` when `d > m`. Let the stored squared values be `D=d^2`, `M=m^2`,
and `W=w^2`. Spheres checks the required sign preconditions `D > M` and
`D + M > W`, then tests `(D + M - W)^2 > 4DM`.

The earlier shortcut `D - M > W` is more aggressive and therefore
approximate: it can prune pages that the exact lower bound retains. It remains
useful as an explicitly approximate policy when the caller accepts that
latency/recall tradeoff, but it must not be labeled exact.
