#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>

#include "benchmark_config.h"

import spheres;

namespace spheres_benchmark {

namespace {

std::uintmax_t directory_size(const std::filesystem::path& path) {
    std::uintmax_t size = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
            size += entry.file_size();
        }
    }
    return size;
}

} // namespace

void run(const BenchmarkConfig& config) {

    std::vector<std::vector<float>> vectors;
    double read_ms = 0.0;
    {
        ScopedTimer timer(read_ms);
        vectors = read_file_as_vectors(
            config.base_vector_path,
            config.n_base_vectors,
            config.vector_dim);
    }
    std::vector<std::size_t> ids(config.n_base_vectors);
    std::iota(ids.begin(), ids.end(), 0);

    double build_ms = 0.0;
    if (!config.reuse_index) {
        std::filesystem::remove_all(config.index_path);
        {
            ScopedTimer timer(build_ms);
            auto database = spheres::create_database<float>(
                config.index_path,
                config.vector_dim,
                spheres::SupportedDistanceFunctions::SquaredL2Distance,
                config.spheres_page_size,
                config.spheres_file_size,
                config.n_threads,
                config.spheres_local_buffer_size);
            spheres::batch_prefill_parallel(database, vectors, ids);
        }
        vectors.clear();
        vectors.shrink_to_fit();
        ids.clear();
        ids.shrink_to_fit();
    }

    decltype(spheres::load_database<float>(config.index_path)) database;
    double load_ms = 0.0;
    {
        ScopedTimer timer(load_ms);
        database = spheres::load_database<float>(
            config.index_path,
            config.n_threads,
            config.spheres_local_buffer_size);
    }

    auto queries = read_file_as_flat_array(
        config.query_vector_path,
        config.n_query_vectors,
        config.vector_dim);
    if (config.engine_mode == "legacy_prune") {
        database->thread_pool->context->prune_check_func =
            [](float dist_to_anchor, float in_page_max, float worst) {
                return dist_to_anchor - in_page_max > worst;
            };
    } else if (config.engine_mode != "exact" && config.engine_mode != "ann") {
        throw std::invalid_argument("unknown Spheres mode: " + config.engine_mode);
    }
    std::vector<std::vector<std::size_t>> results(config.n_query_vectors);
    double search_ms = 0.0;
    {
        ScopedTimer timer(search_ms);
        for (std::size_t query = 0; query < config.n_query_vectors; ++query) {
            const auto nearest = spheres::find_k_nearest(
                database,
                std::span<float>(queries.data() + query * config.vector_dim, config.vector_dim),
                config.k,
                config.spheres_n_pages_to_probe,
                config.spheres_morsel_step);
            auto& row = results[query];
            row.reserve(nearest.size());
            for (const auto& [distance, id] : nearest) {
                static_cast<void>(distance);
                row.push_back(id);
            }
        }
    }

    const auto ground_truth = read_gt_vectors(
        config.gt_vector_path,
        config.n_query_vectors,
        config.k_in_gt);
    const auto index_bytes = directory_size(config.index_path);
    database.reset();
    const auto base = read_file_as_flat_array(
        config.base_vector_path,
        config.n_base_vectors,
        config.vector_dim);
    const auto average_distance = avg_squared_l2_distance(
        std::span<const float>(queries.data(), queries.size()),
        std::span<const float>(base.data(), base.size()),
        config.vector_dim,
        results,
        ground_truth,
        config.k);
    write_json(config.metrics_path, {
        {"engine", quote("spheres")},
        {"mode", quote(config.engine_mode)},
        {"count", std::to_string(config.n_base_vectors)},
        {"dim", std::to_string(config.vector_dim)},
        {"query_count", std::to_string(config.n_query_vectors)},
        {"k", std::to_string(config.k)},
        {"threads", std::to_string(config.n_threads)},
        {"query_concurrency", "1"},
        {"distance", quote("squared_l2")},
        {"read_boundary", quote("read all base vectors into application memory")},
        {"build_boundary", quote("create database and parallel batch prefill")},
        {"read_ms", std::to_string(read_ms)},
        {"build_ms", std::to_string(build_ms)},
        {"load_ms", std::to_string(load_ms)},
        {"search_ms", std::to_string(search_ms)},
        {"avg_query_ms", std::to_string(search_ms / config.n_query_vectors)},
        {"recall_at_k", std::to_string(recall(results, ground_truth, config.k))},
        {"avg_squared_l2_gt", std::to_string(average_distance.second)},
        {"avg_squared_l2", std::to_string(average_distance.first)},
        {"index_bytes", std::to_string(index_bytes)},
        {"page_size", std::to_string(config.spheres_page_size)},
        {"file_size", std::to_string(config.spheres_file_size)},
        {"pages_to_probe", std::to_string(config.spheres_n_pages_to_probe)},
        {"morsel_step", std::to_string(config.spheres_morsel_step)}
    });
}

} // namespace spheres_benchmark
