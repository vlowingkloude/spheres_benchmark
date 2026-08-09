module;

#include <cstddef>
#include <utility>
#include <memory>
#include <fstream>
#include <span>

export module builtin_qwen_model;

import builtin_qwen_model_generated;
import memorymap;

export namespace spheres::builtin_qwen_embedding::qwen_model {

    constexpr std::size_t embedding_dim {1024};
    constexpr std::size_t norm_dim {1024};

    struct QwenWeights {
        spheres::memorymap::MappedFile mapped_file;
        std::byte *base_ptr;
        std::size_t header_size;
        std::byte *weight_begin;
        generated_model::Config config {};

        QwenWeights() = delete;

        explicit QwenWeights(const std::string& filename) : mapped_file(filename) {
            base_ptr = mapped_file.addr;
            header_size = *((std::bit_cast<std::size_t *, std::byte *>(base_ptr)));
            weight_begin = base_ptr + header_size + sizeof(std::size_t);
        }
    };

    std::span<uint16_t> get_weights_of_layer(const QwenWeights& w, const std::string& name, const std::size_t layer_index, const std::size_t size) {
        return {std::bit_cast<uint16_t *>(w.weight_begin + generated_model::qwen_weight_offsets.at(name)[layer_index]), size};
    }

    std::span<uint16_t> get_weights_of_layer(const QwenWeights& w, const std::string& name, const std::size_t layer_index, const std::size_t l1, const std::size_t l2) {
        return {std::bit_cast<uint16_t *>(w.weight_begin + generated_model::qwen_weight_offsets.at(name)[layer_index]), l1 * l2};
    }

    std::span<uint16_t> get_embedding_of_token(const QwenWeights& w, const int token_id) {
        return {std::bit_cast<uint16_t *>(w.weight_begin + generated_model::embed_tokens_weight_offset) + token_id * embedding_dim, embedding_dim};
    }

    std::span<uint16_t> get_norm_weights(const QwenWeights& w) {
        return {std::bit_cast<uint16_t *>(w.weight_begin + generated_model::norm_weight_offset), norm_dim};
    }

}
