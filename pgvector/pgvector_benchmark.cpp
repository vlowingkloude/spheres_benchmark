#include <arpa/inet.h>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <libpq-fe.h>

#include "benchmark_config.h"
#include "fvecs_stream.h"

namespace pgvector_benchmark {

namespace {

struct ConnectionDeleter {
    void operator()(PGconn* connection) const { PQfinish(connection); }
};

struct ResultDeleter {
    void operator()(PGresult* result) const { PQclear(result); }
};

using Connection = std::unique_ptr<PGconn, ConnectionDeleter>;
using PgResult = std::unique_ptr<PGresult, ResultDeleter>;

PgResult execute(PGconn* connection, const std::string& sql) {
    PgResult result(PQexec(connection, sql.c_str()));
    if (!result || (PQresultStatus(result.get()) != PGRES_COMMAND_OK &&
                    PQresultStatus(result.get()) != PGRES_TUPLES_OK)) {
        throw std::runtime_error(PQerrorMessage(connection));
    }
    return result;
}

void append_u16(std::vector<char>& output, const std::uint16_t value) {
    const std::uint16_t network = htons(value);
    const auto* bytes = reinterpret_cast<const char*>(&network);
    output.insert(output.end(), bytes, bytes + sizeof(network));
}

void append_u32(std::vector<char>& output, const std::uint32_t value) {
    const std::uint32_t network = htonl(value);
    const auto* bytes = reinterpret_cast<const char*>(&network);
    output.insert(output.end(), bytes, bytes + sizeof(network));
}

void append_u64(std::vector<char>& output, const std::uint64_t value) {
    append_u32(output, static_cast<std::uint32_t>(value >> 32));
    append_u32(output, static_cast<std::uint32_t>(value));
}

void copy_vectors(PGconn* connection, const BenchmarkConfig& config) {
    PgResult start(PQexec(
        connection,
        "COPY spheres_benchmark_vectors(id, embedding) FROM STDIN WITH (FORMAT binary)"));
    if (!start || PQresultStatus(start.get()) != PGRES_COPY_IN) {
        throw std::runtime_error(PQerrorMessage(connection));
    }

    std::vector<char> header;
    constexpr char signature[] = "PGCOPY\n\377\r\n\0";
    header.insert(header.end(), signature, signature + 11);
    append_u32(header, 0);
    append_u32(header, 0);
    if (PQputCopyData(connection, header.data(), static_cast<int>(header.size())) != 1) {
        throw std::runtime_error(PQerrorMessage(connection));
    }

    constexpr std::size_t batch_rows = 1'000;
    FvecsBatchReader reader(config.base_vector_path, config.vector_dim);
    std::size_t copied = 0;
    while (copied < config.n_base_vectors) {
        const std::size_t rows = std::min(batch_rows, config.n_base_vectors - copied);
        const auto batch = reader.read(rows);
        std::vector<char> payload;
        payload.reserve(rows * (18 + config.vector_dim * sizeof(float)));
        for (std::size_t row = 0; row < rows; ++row) {
            append_u16(payload, 2);
            append_u32(payload, 8);
            append_u64(payload, copied + row);

            append_u32(payload, static_cast<std::uint32_t>(4 + config.vector_dim * sizeof(float)));
            append_u16(payload, static_cast<std::uint16_t>(config.vector_dim));
            append_u16(payload, 0);
            for (std::size_t column = 0; column < config.vector_dim; ++column) {
                append_u32(payload, std::bit_cast<std::uint32_t>(
                    batch[row * config.vector_dim + column]));
            }
        }
        if (PQputCopyData(connection, payload.data(), static_cast<int>(payload.size())) != 1) {
            throw std::runtime_error(PQerrorMessage(connection));
        }
        copied += rows;
    }

    const char trailer[] = {static_cast<char>(0xff), static_cast<char>(0xff)};
    if (PQputCopyData(connection, trailer, 2) != 1 || PQputCopyEnd(connection, nullptr) != 1) {
        throw std::runtime_error(PQerrorMessage(connection));
    }
    while (PGresult* raw_result = PQgetResult(connection)) {
        PgResult result(raw_result);
        if (PQresultStatus(result.get()) != PGRES_COMMAND_OK) {
            throw std::runtime_error(PQerrorMessage(connection));
        }
    }
}

std::string vector_literal(const float* vector, const std::size_t dim) {
    std::ostringstream output;
    output.precision(9);
    output << '[';
    for (std::size_t i = 0; i < dim; ++i) {
        if (i != 0) {
            output << ',';
        }
        output << vector[i];
    }
    output << ']';
    return output.str();
}

} // namespace

void run(const BenchmarkConfig& config) {
    const std::string connection_string =
        "host=" + config.pg_host +
        " port=" + std::to_string(config.pg_port) +
        " user=" + config.pg_user +
        " password=" + config.pg_password +
        " dbname=" + config.pg_database;

    double load_ms = 0.0;
    Connection connection;
    {
        ScopedTimer timer(load_ms);
        connection.reset(PQconnectdb(connection_string.c_str()));
        if (!connection || PQstatus(connection.get()) != CONNECTION_OK) {
            throw std::runtime_error(connection ? PQerrorMessage(connection.get()) : "PQconnectdb failed");
        }
    }

    execute(connection.get(), "CREATE EXTENSION IF NOT EXISTS vector");
    execute(connection.get(), "DROP TABLE IF EXISTS spheres_benchmark_vectors");
    execute(connection.get(),
        "CREATE TABLE spheres_benchmark_vectors(id bigint PRIMARY KEY, embedding vector(" +
        std::to_string(config.vector_dim) + "))");

    double ingest_ms = 0.0;
    {
        ScopedTimer timer(ingest_ms);
        copy_vectors(connection.get(), config);
    }

    double build_ms = 0.0;
    if (config.engine_mode == "hnsw") {
        execute(connection.get(),
            "SET maintenance_work_mem = '" + config.pg_maintenance_work_mem + "'");
        ScopedTimer timer(build_ms);
        execute(connection.get(),
            "CREATE INDEX spheres_benchmark_hnsw ON spheres_benchmark_vectors "
            "USING hnsw (embedding vector_l2_ops) WITH (m = " +
            std::to_string(config.pg_hnsw_m) + ", ef_construction = " +
            std::to_string(config.pg_hnsw_ef_construction) + ")");
    } else if (config.engine_mode != "flat") {
        throw std::invalid_argument("unknown pgvector mode: " + config.engine_mode);
    }

    execute(connection.get(), "SET hnsw.ef_search = " + std::to_string(config.pg_hnsw_ef_search));
    execute(connection.get(), "ANALYZE spheres_benchmark_vectors");

    const auto queries = read_file_as_flat_array(
        config.query_vector_path,
        config.n_query_vectors,
        config.vector_dim);
    std::vector<std::string> query_parameters;
    query_parameters.reserve(config.n_query_vectors);
    for (std::size_t query = 0; query < config.n_query_vectors; ++query) {
        query_parameters.push_back(vector_literal(
            queries.data() + query * config.vector_dim,
            config.vector_dim));
    }

    const std::string query_sql =
        "SELECT id FROM spheres_benchmark_vectors ORDER BY embedding <-> $1::vector LIMIT " +
        std::to_string(config.k);
    if (config.engine_mode == "hnsw") {
        const auto plan = execute(
            connection.get(),
            "EXPLAIN SELECT id FROM spheres_benchmark_vectors ORDER BY embedding <-> '" +
            query_parameters.front() + "'::vector LIMIT " + std::to_string(config.k));
        std::string plan_text;
        for (int row = 0; row < PQntuples(plan.get()); ++row) {
            plan_text += PQgetvalue(plan.get(), row, 0);
            plan_text.push_back('\n');
        }
        if (plan_text.find("spheres_benchmark_hnsw") == std::string::npos) {
            throw std::runtime_error("PostgreSQL did not select the HNSW index:\n" + plan_text);
        }
    }
    std::vector<std::vector<std::size_t>> results(config.n_query_vectors);
    double search_ms = 0.0;
    {
        ScopedTimer timer(search_ms);
        for (std::size_t query = 0; query < config.n_query_vectors; ++query) {
            const char* values[] = {query_parameters[query].c_str()};
            PgResult result(PQexecParams(
                connection.get(), query_sql.c_str(), 1, nullptr, values, nullptr, nullptr, 0));
            if (!result || PQresultStatus(result.get()) != PGRES_TUPLES_OK) {
                throw std::runtime_error(PQerrorMessage(connection.get()));
            }
            auto& ids = results[query];
            ids.reserve(config.k);
            for (int row = 0; row < PQntuples(result.get()); ++row) {
                ids.push_back(std::stoull(PQgetvalue(result.get(), row, 0)));
            }
            if (ids.size() != config.k) {
                throw std::runtime_error("pgvector returned fewer than k neighbors");
            }
        }
    }

    const auto ground_truth = read_gt_vectors(
        config.gt_vector_path,
        config.n_query_vectors,
        config.k_in_gt);
    const auto size_result = execute(
        connection.get(),
        "SELECT pg_total_relation_size('spheres_benchmark_vectors')");

    write_json(config.metrics_path, {
        {"engine", quote("pgvector")},
        {"mode", quote(config.engine_mode)},
        {"count", std::to_string(config.n_base_vectors)},
        {"dim", std::to_string(config.vector_dim)},
        {"query_count", std::to_string(config.n_query_vectors)},
        {"k", std::to_string(config.k)},
        {"client_threads", std::to_string(config.n_threads)},
        {"query_concurrency", "1"},
        {"deployment", quote("remote_client_server")},
        {"distance", quote("l2_ordering_equivalent_to_squared_l2")},
        {"ingest_boundary", quote("stream fvecs and send binary COPY over network")},
        {"build_boundary", quote("create HNSW index on server")},
        {"connect_ms", std::to_string(load_ms)},
        {"ingest_ms", std::to_string(ingest_ms)},
        {"build_ms", std::to_string(build_ms)},
        {"search_ms", std::to_string(search_ms)},
        {"avg_query_ms", std::to_string(search_ms / config.n_query_vectors)},
        {"recall_at_k", std::to_string(recall(results, ground_truth, config.k))},
        {"relation_bytes", PQgetvalue(size_result.get(), 0, 0)},
        {"m", std::to_string(config.pg_hnsw_m)},
        {"ef_construction", std::to_string(config.pg_hnsw_ef_construction)},
        {"ef_search", std::to_string(config.pg_hnsw_ef_search)},
        {"maintenance_work_mem", quote(config.pg_maintenance_work_mem)}
    });
}

} // namespace pgvector_benchmark
