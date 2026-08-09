#include <algorithm>
#include <filesystem>
#include <memory>
#include <queue>
#include <vector>

#include <hnswlib/hnswlib.h>
#include <omp.h>

#include "benchmark_config.h"
#include "fvecs_stream.h"

namespace hnswlib_benchmark {

void run(const BenchmarkConfig& config) {
    hnswlib::L2Space space(config.vector_dim);
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    double build_ms = 0.0;
    {
        ScopedTimer timer(build_ms);
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            &space,
            config.n_base_vectors,
            config.hnswlib_m,
            config.hnswlib_ef_construction);
        index->setEf(config.hnswlib_ef_search);

        constexpr std::size_t batch_size = 10'000;
        FvecsBatchReader reader(config.base_vector_path, config.vector_dim);
        std::size_t added = 0;
        while (added < config.n_base_vectors) {
            const std::size_t rows = std::min(batch_size, config.n_base_vectors - added);
            const auto batch = reader.read(rows);
#pragma omp parallel for schedule(static) num_threads(config.n_threads)
            for (std::size_t row = 0; row < rows; ++row) {
                index->addPoint(
                    batch.data() + row * config.vector_dim,
                    added + row);
            }
            added += rows;
        }
    }

    double save_ms = 0.0;
    {
        ScopedTimer timer(save_ms);
        index->saveIndex(config.index_path);
    }
    index.reset();

    double load_ms = 0.0;
    {
        ScopedTimer timer(load_ms);
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            &space,
            config.index_path,
            false,
            config.n_base_vectors);
        index->setEf(config.hnswlib_ef_search);
    }

    const auto queries = read_file_as_flat_array(
        config.query_vector_path,
        config.n_query_vectors,
        config.vector_dim);
    std::vector<std::vector<std::size_t>> results(config.n_query_vectors);

    double search_ms = 0.0;
    {
        ScopedTimer timer(search_ms);
        for (std::size_t query = 0; query < config.n_query_vectors; ++query) {
            auto heap = index->searchKnn(
                queries.data() + query * config.vector_dim,
                config.k);
            auto& row = results[query];
            row.resize(heap.size());
            for (std::size_t rank = heap.size(); rank > 0; --rank) {
                row[rank - 1] = heap.top().second;
                heap.pop();
            }
        }
    }

    const auto ground_truth = read_gt_vectors(
        config.gt_vector_path,
        config.n_query_vectors,
        config.k_in_gt);
    write_json(config.metrics_path, {
        {"engine", quote("hnswlib")},
        {"mode", quote("hnsw")},
        {"count", std::to_string(config.n_base_vectors)},
        {"dim", std::to_string(config.vector_dim)},
        {"query_count", std::to_string(config.n_query_vectors)},
        {"k", std::to_string(config.k)},
        {"threads", std::to_string(config.n_threads)},
        {"query_concurrency", "1"},
        {"distance", quote("squared_l2")},
        {"build_boundary", quote("stream base vectors and construct HNSW")},
        {"build_ms", std::to_string(build_ms)},
        {"save_ms", std::to_string(save_ms)},
        {"load_ms", std::to_string(load_ms)},
        {"search_ms", std::to_string(search_ms)},
        {"avg_query_ms", std::to_string(search_ms / config.n_query_vectors)},
        {"recall_at_k", std::to_string(recall(results, ground_truth, config.k))},
        {"index_bytes", std::to_string(std::filesystem::file_size(config.index_path))},
        {"m", std::to_string(config.hnswlib_m)},
        {"ef_construction", std::to_string(config.hnswlib_ef_construction)},
        {"ef_search", std::to_string(config.hnswlib_ef_search)}
    });
}

} // namespace hnswlib_benchmark
