#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <duckdb.hpp>
#include <vss_extension.hpp>

#include "benchmark_config.h"
#include "fvecs_stream.h"

namespace duckdb_benchmark {

namespace {

void require_success(const auto& result) {
    if (!result || result->HasError()) {
        throw std::runtime_error(result ? result->GetError() : "DuckDB returned no result");
    }
}

duckdb::Value make_array(const float* values, const std::size_t dim) {
    duckdb::vector<duckdb::Value> elements;
    elements.reserve(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        elements.emplace_back(values[i]);
    }
    return duckdb::Value::ARRAY(duckdb::LogicalType::FLOAT, std::move(elements));
}

std::string make_array_literal(const float* values, const std::size_t dim) {
    std::ostringstream out;
    out.precision(std::numeric_limits<float>::max_digits10);
    out << '[';
    for (std::size_t i = 0; i < dim; ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    out << "]::FLOAT[" << dim << ']';
    return out.str();
}

void initialize_connection(duckdb::Connection& connection, const BenchmarkConfig& config) {
    require_success(connection.Query("LOAD vss"));
    require_success(connection.Query(
        "SET threads = " + std::to_string(config.n_threads)));
    require_success(connection.Query(
        "SET hnsw_enable_experimental_persistence = true"));
}

} // namespace

void run(const BenchmarkConfig& config) {
    std::filesystem::remove(config.index_path);
    std::filesystem::remove(config.index_path + ".wal");

    auto database = std::make_unique<duckdb::DuckDB>(config.index_path);
    auto connection = std::make_unique<duckdb::Connection>(*database);
    initialize_connection(*connection, config);

    require_success(connection->Query(
        "CREATE TABLE embeddings(id UBIGINT, vec FLOAT[" +
        std::to_string(config.vector_dim) + "])"));

    double ingest_ms = 0.0;
    {
        ScopedTimer timer(ingest_ms);
        duckdb::Appender appender(*connection, "embeddings");
        constexpr std::size_t batch_size = 10'000;
        FvecsBatchReader reader(config.base_vector_path, config.vector_dim);
        std::size_t added = 0;
        while (added < config.n_base_vectors) {
            const std::size_t rows = std::min(batch_size, config.n_base_vectors - added);
            const auto batch = reader.read(rows);
            for (std::size_t row = 0; row < rows; ++row) {
                appender.BeginRow();
                appender.Append<std::uint64_t>(added + row);
                appender.Append<duckdb::Value>(make_array(
                    batch.data() + row * config.vector_dim,
                    config.vector_dim));
                appender.EndRow();
            }
            added += rows;
        }
        appender.Close();
    }

    double build_ms = 0.0;
    {
        ScopedTimer timer(build_ms);
        require_success(connection->Query(
            "CREATE INDEX embeddings_hnsw ON embeddings USING HNSW(vec) "
            "WITH (metric = 'l2sq', M = " + std::to_string(config.duckdb_hnsw_m) +
            ", ef_construction = " + std::to_string(config.duckdb_hnsw_ef_construction) +
            ", ef_search = " + std::to_string(config.duckdb_hnsw_ef_search) + ")"));
        require_success(connection->Query("CHECKPOINT"));
    }

    connection.reset();
    database.reset();

    double load_ms = 0.0;
    {
        ScopedTimer timer(load_ms);
        database = std::make_unique<duckdb::DuckDB>(config.index_path);
        connection = std::make_unique<duckdb::Connection>(*database);
        initialize_connection(*connection, config);
    }

    const auto queries = read_file_as_flat_array(
        config.query_vector_path,
        config.n_query_vectors,
        config.vector_dim);
    const auto explain = connection->Query(
        "EXPLAIN SELECT id FROM embeddings ORDER BY array_distance(vec, " +
        make_array_literal(queries.data(), config.vector_dim) + ") LIMIT " +
        std::to_string(config.k));
    require_success(explain);
    if (explain->ToString().find("HNSW_INDEX_SCAN") == std::string::npos) {
        throw std::runtime_error(
            "DuckDB did not select HNSW_INDEX_SCAN for the benchmark query:\n" +
            explain->ToString());
    }
    auto prepared = connection->Prepare(
        "SELECT id FROM embeddings "
        "ORDER BY array_distance(vec, ?::FLOAT[" + std::to_string(config.vector_dim) + "]) "
        "LIMIT " + std::to_string(config.k));
    if (!prepared || prepared->HasError()) {
        throw std::runtime_error(prepared ? prepared->GetError() : "failed to prepare DuckDB query");
    }

    std::vector<std::vector<std::size_t>> results(config.n_query_vectors);
    double search_ms = 0.0;
    {
        ScopedTimer timer(search_ms);
        for (std::size_t query = 0; query < config.n_query_vectors; ++query) {
            auto result = prepared->Execute(make_array(
                queries.data() + query * config.vector_dim,
                config.vector_dim));
            require_success(result);
            auto& row_ids = results[query];
            row_ids.reserve(config.k);
            for (const auto& row : *result) {
                row_ids.push_back(row.GetValue<std::uint64_t>(0));
            }
            if (row_ids.size() != config.k) {
                throw std::runtime_error("DuckDB returned fewer than k neighbors");
            }
        }
    }

    const auto ground_truth = read_gt_vectors(
        config.gt_vector_path,
        config.n_query_vectors,
        config.k_in_gt);
    write_json(config.metrics_path, {
        {"engine", quote("duckdb_vss")},
        {"mode", quote("hnsw")},
        {"count", std::to_string(config.n_base_vectors)},
        {"dim", std::to_string(config.vector_dim)},
        {"query_count", std::to_string(config.n_query_vectors)},
        {"k", std::to_string(config.k)},
        {"threads", std::to_string(config.n_threads)},
        {"query_concurrency", "1"},
        {"distance", quote("squared_l2")},
        {"ingest_boundary", quote("stream fvecs and append rows")},
        {"build_boundary", quote("create persistent HNSW index and checkpoint")},
        {"ingest_ms", std::to_string(ingest_ms)},
        {"build_ms", std::to_string(build_ms)},
        {"load_ms", std::to_string(load_ms)},
        {"search_ms", std::to_string(search_ms)},
        {"avg_query_ms", std::to_string(search_ms / config.n_query_vectors)},
        {"recall_at_k", std::to_string(recall(results, ground_truth, config.k))},
        {"database_bytes", std::to_string(std::filesystem::file_size(config.index_path))},
        {"m", std::to_string(config.duckdb_hnsw_m)},
        {"ef_construction", std::to_string(config.duckdb_hnsw_ef_construction)},
        {"ef_search", std::to_string(config.duckdb_hnsw_ef_search)}
    });
}

} // namespace duckdb_benchmark
