#include <algorithm>
#include <cmath>
#include <queue>
#include <span>
#include <utility>
#include <vector>

#include "benchmark_config.h"

namespace baseline {

namespace {

using HeapEntry = std::pair<float, std::size_t>;

float squared_l2(const std::span<const float> x, const std::span<const float> y) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float difference = x[i] - y[i];
        sum += difference * difference;
    }
    return sum;
}

} // namespace

void run(const BenchmarkConfig& config) {
    std::vector<float> vectors;
    double read_base_time = 0.0;
    {
        ScopedTimer timer(read_base_time);
        vectors = read_file_as_flat_array(
            config.base_vector_path,
            config.n_base_vectors,
            config.vector_dim);
    }

    const auto queries = read_file_as_flat_array(
        config.query_vector_path,
        config.n_query_vectors,
        config.vector_dim);

    std::vector<std::vector<std::size_t>> results(config.n_query_vectors);
    double query_time = 0.0;
    {
        ScopedTimer timer(query_time);
        for (std::size_t query_index = 0;
             query_index < config.n_query_vectors;
             ++query_index) {
            std::priority_queue<HeapEntry> heap;
            const std::span<const float> query(
                queries.data() + query_index * config.vector_dim,
                config.vector_dim);

            for (std::size_t vector_index = 0;
                 vector_index < config.n_base_vectors;
                 ++vector_index) {
                const std::span<const float> vector(
                    vectors.data() + vector_index * config.vector_dim,
                    config.vector_dim);
                const float distance = squared_l2(vector, query);

                if (heap.size() < config.k) {
                    heap.emplace(distance, vector_index);
                } else if (distance < heap.top().first) {
                    heap.pop();
                    heap.emplace(distance, vector_index);
                }
            }

            auto& query_results = results[query_index];
            query_results.resize(heap.size());
            for (std::size_t rank = heap.size(); rank > 0; --rank) {
                query_results[rank - 1] = heap.top().second;
                heap.pop();
            }
        }
    }

    const auto ground_truth = read_gt_vectors(
        config.gt_vector_path,
        config.n_query_vectors,
        config.k_in_gt);
    const auto average_distance = avg_squared_l2_distance(
        queries,
        vectors,
        config.vector_dim,
        results,
        ground_truth,
        config.k);

    write_json(config.metrics_path, {
        {"engine", quote(config.engine_name)},
        {"mode", quote(config.engine_mode)},
        {"vectors_path", quote(config.dataset_path)},
        {"count", std::to_string(config.n_base_vectors)},
        {"dim", std::to_string(config.vector_dim)},
        {"query_count", std::to_string(config.n_query_vectors)},
        {"k", std::to_string(config.k)},
        {"threads", "1"},
        {"read_ms", std::to_string(read_base_time)},
        {"prepare_ms", "0.0"},
        {"build_ms", "0.0"},
        {"search_ms", std::to_string(query_time)},
        {"avg_query_ms", std::to_string(
            query_time / static_cast<double>(config.n_query_vectors))},
        {"recall_at_k", std::to_string(recall(results, ground_truth, config.k))},
        {"avg_squared_l2_gt", std::to_string(average_distance.second)},
        {"avg_squared_l2", std::to_string(average_distance.first)}
    });
}

} // namespace baseline
