module;

#include <string_view>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <span>
#include <array>
#include <optional>
#include <cmath>
#include <vector>
#include <ranges>

export module builtin_qwen_embedding;

import array_view;
import builtin_qwen_tokenizer;
import builtin_qwen_model;
import builtin_qwen_model_generated;

export namespace spheres::builtin_qwen_embedding::tensor {

    struct Tensor {
        const std::size_t size;
        const bool is_owner;
        float *data;
        std::size_t dim;
        std::array<std::size_t, 4> shape = {0, 1, 1, 1};

        Tensor() : size(0), is_owner(false), data(nullptr), dim(0) {}

        explicit Tensor(const std::size_t size) : size(size), is_owner(true) {
            data = std::bit_cast<float*>(malloc(size * sizeof(float)));
            shape[0] = size;
            dim = 1;
        }

        Tensor(const std::size_t size, float* p, const std::size_t dim,
            const std::size_t s1 = 0, const std::size_t s2 = 1, const std::size_t s3 = 1, const std::size_t s4 = 1)
            : size(size), is_owner(false), data(p), dim(dim) {
            shape[0] = s1;
            shape[1] = s2;
            shape[2] = s3;
            shape[3] = s4;
        }

        Tensor operator[](const std::size_t index) {
            const auto n_elements= shape[1] * shape[2] * shape[3];
            const auto offset = index * n_elements;
            if (data == nullptr || shape.empty() || offset >= size) {
                return {};
            }
            float *p = data + offset;
            constexpr std::array<std::size_t, 5> lut {0, 1, 1, 2, 3};
            return {n_elements, p, lut[dim], shape[1], shape[2], shape[3], 1};
        }

        ~Tensor() {
            // It is safe here, since the lifetime of the owner will only be freed after a forward pass and will anyway longer than slices in this module
            if (data != nullptr && is_owner) {
                free(data);
                data = nullptr;
            }
        }
    };

    inline void flatten(Tensor& tensor) {
        tensor.shape[0] = tensor.size;
        tensor.shape[1] = 1;
        tensor.shape[2] = 1;
        tensor.shape[3] = 1;
        tensor.dim = 1;
    }

    inline void reshape(Tensor& tensor, const std::size_t s1, const std::size_t s2) {
        if (tensor.data != nullptr && s1 * s2 == tensor.size) {
            tensor.shape[0] = s1;
            tensor.shape[1] = s2;
            tensor.shape[2] = 1;
            tensor.shape[3] = 1;
            tensor.dim = 2;
        }
    }

    inline void reshape(Tensor& tensor, const std::size_t s1, const std::size_t s2, const std::size_t s3) {
        if (tensor.data != nullptr && s1 * s2 * s3 == tensor.size) {
            tensor.shape[0] = s1;
            tensor.shape[1] = s2;
            tensor.shape[2] = s3;
            tensor.shape[3] = 1;
            tensor.dim = 3;
        }
    }

    inline void set_scalar(const Tensor& tensor, const std::size_t index, const float value) {
        if (tensor.dim == 1) {
            tensor.data[index] = value;
        }
    }

    inline std::optional<float> get_scalar(const Tensor& tensor, const std::size_t index) {
        if (tensor.dim == 1) {
            return tensor.data[index];
        }
        return std::nullopt;
    }

    inline std::span<float> get_span_view(const Tensor& t) {
        return {t.data, t.size};
    }

    void copy(const Tensor& dst, std::span<float> source) {
        const auto size = dst.shape[0] * dst.shape[1] * dst.shape[2] * dst.shape[3];
        memcpy(dst.data, source.data(), size * sizeof(float));
    }

}

namespace spheres::builtin_qwen_embedding::math {

    inline float float_bf16_scalar_add_scalar(const float x, const uint16_t y) {
        const uint32_t iy = static_cast<uint32_t>(y) << 16;
        const auto fy = std::bit_cast<float, uint32_t>(iy);
        return x + fy;
    }

    inline float float_bf16_scalar_product_scalar(const float x, const uint16_t y) {
        const uint32_t iy = static_cast<uint32_t>(y) << 16;
        const auto fy = std::bit_cast<float, uint32_t>(iy);
        return x * fy;
    }

    inline float bf16_bf16_scalar_add_scalar(const uint16_t x, const uint16_t y) {
        const uint32_t ix = static_cast<uint32_t>(x) << 16;
        const auto fx = std::bit_cast<float, uint32_t>(ix);
        return float_bf16_scalar_add_scalar(fx, y);
    }

    inline float bf16_bf16_scalar_product_scalar(const uint16_t x, const uint16_t y) {
        const uint32_t ix = static_cast<uint32_t>(x) << 16;
        const auto fx = std::bit_cast<float, uint32_t>(ix);
        return float_bf16_scalar_product_scalar(fx, y);
    }

    float dot_product_float_bf16_vector(const std::span<float> x, const std::span<uint16_t> y) {
        float result = 0.0f;
        float xy0 = 0, xy1, xy2, xy3, xy4, xy5, xy6, xy7;
        xy0 = xy1 = xy2 = xy3 = xy4 = xy5 = xy6 = xy7 = 0;
        std::size_t i = 0;
        for (; i + 7 < y.size(); i += 8) {
            const uint32_t iy0 = static_cast<uint32_t>(y[i + 0]) << 16;
            const uint32_t iy1 = static_cast<uint32_t>(y[i + 1]) << 16;
            const uint32_t iy2 = static_cast<uint32_t>(y[i + 2]) << 16;
            const uint32_t iy3 = static_cast<uint32_t>(y[i + 3]) << 16;
            const uint32_t iy4 = static_cast<uint32_t>(y[i + 4]) << 16;
            const uint32_t iy5 = static_cast<uint32_t>(y[i + 5]) << 16;
            const uint32_t iy6 = static_cast<uint32_t>(y[i + 6]) << 16;
            const uint32_t iy7 = static_cast<uint32_t>(y[i + 7]) << 16;
            const auto fy0 = std::bit_cast<float, uint32_t>(iy0);
            const auto fy1 = std::bit_cast<float, uint32_t>(iy1);
            const auto fy2 = std::bit_cast<float, uint32_t>(iy2);
            const auto fy3 = std::bit_cast<float, uint32_t>(iy3);
            const auto fy4 = std::bit_cast<float, uint32_t>(iy4);
            const auto fy5 = std::bit_cast<float, uint32_t>(iy5);
            const auto fy6 = std::bit_cast<float, uint32_t>(iy6);
            const auto fy7 = std::bit_cast<float, uint32_t>(iy7);
            xy0 = x[i + 0] * fy0 + xy0;
            xy1 = x[i + 1] * fy1 + xy1;
            xy2 = x[i + 2] * fy2 + xy2;
            xy3 = x[i + 3] * fy3 + xy3;
            xy4 = x[i + 4] * fy4 + xy4;
            xy5 = x[i + 5] * fy5 + xy5;
            xy6 = x[i + 6] * fy6 + xy6;
            xy7 = x[i + 7] * fy7 + xy7;
        }
        result = ((xy0 + xy4) + (xy1 + xy5)) + ((xy2 + xy6) + (xy3 + xy7));
        if (i != y.size()) [[unlikely]]  {
            for (; i < y.size(); ++i) {
                result += float_bf16_scalar_product_scalar(x[i], y[i]);
            }
        }
        return result;
    }

    float dot_product_float_float_vector(const std::span<float> x, const std::span<float> y) {
        float dot_product {0};
        std::size_t i;
        float xy0 = 0, xy1, xy2, xy3, xy4, xy5, xy6, xy7;
        xy0 = xy1 = xy2 = xy3 = xy4 = xy5 = xy6 = xy7 = 0;
        for (i = 0; i + 7 < x.size(); i += 8) {
            xy0 = x[i + 0] * y[i + 0] + xy0;
            xy1 = x[i + 1] * y[i + 1] + xy1;
            xy2 = x[i + 2] * y[i + 2] + xy2;
            xy3 = x[i + 3] * y[i + 3] + xy3;
            xy4 = x[i + 4] * y[i + 4] + xy4;
            xy5 = x[i + 5] * y[i + 5] + xy5;
            xy6 = x[i + 6] * y[i + 6] + xy6;
            xy7 = x[i + 7] * y[i + 7] + xy7;
        }
        dot_product = ((xy0 + xy4) + (xy1 + xy5)) + ((xy2 + xy6) + (xy3 + xy7));
        if (i != x.size()) [[unlikely]] {
            for (;i < x.size(); ++i) {
                dot_product = x[i] * y[i] + dot_product;
            }
        }
        return dot_product;
    }

    void rms_norm_float_bf16_weights(std::span<float> x, std::span<uint16_t> weights, const tensor::Tensor &result, float eps) {
        const float mean = dot_product_float_float_vector(x, x) / static_cast<float>(x.size());
        const float scale = 1.0f / sqrtf(mean + eps);
        for (std::size_t i = 0; i < x.size(); ++i) {
            tensor::set_scalar(result, i, float_bf16_scalar_add_scalar(1.0f, weights[i]) * scale * x[i]);
        }
    }

    void matrix_multiply_vector_bf16_f32(spheres::array_view::ArrayView<uint16_t>& matrix, std::span<float> x, const tensor::Tensor &result) {
        for (std::size_t i = 0; i < matrix.s1; i++) {
            tensor::set_scalar(result, i, dot_product_float_bf16_vector(x, matrix[i]));
        }
    }

    void scale_vector_float(const tensor::Tensor &t, const float scale) {
        for (std::size_t i = 0; i < t.size; i++) {
            tensor::set_scalar(t, i, scale * tensor::get_scalar(t, i).value());
        }
    }

    float max_float_array(const tensor::Tensor &t) {
        float max = -(__builtin_inff());
        for (std::size_t i = 0; i < t.size; i++) {
            max = max > tensor::get_scalar(t, i).value() ? max : tensor::get_scalar(t, i).value();
        }
        return max;
    }

    void softmax_vector_float(const tensor::Tensor &t) {
        const float max = max_float_array(t);
        float sum = 0.0f;
        for (std::size_t i = 0; i < t.size; i++) {
            const float v = expf(tensor::get_scalar(t, i).value() - max);
            sum += v;
            tensor::set_scalar(t, i, v);
        }
        sum = 1.0f / sum;
        scale_vector_float(t, sum);
    }

    void element_wise_fused_scalar_multiply_add_float_vector_overwrite_dst(const tensor::Tensor &t, const float y, const tensor::Tensor &dst) {
        for (std::size_t i = 0; i < t.size; i++) {
            tensor::set_scalar(dst, i, tensor::get_scalar(t, i).value() * y);
        }
    }

    void element_wise_fused_scalar_multiply_add_float(const tensor::Tensor &t, const float y, const tensor::Tensor &dst) {
        for (std::size_t i = 0; i < t.size; i++) {
            tensor::set_scalar(dst, i, tensor::get_scalar(t, i).value() * y + tensor::get_scalar(dst, i).value());
        }
    }

    float sigmoid_scalar(const float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    void swiglu_float_vector(const tensor::Tensor &x, const tensor::Tensor &y, const tensor::Tensor &result) {
        for (std::size_t i = 0; i < x.size; i++) {
            const float xi = tensor::get_scalar(x, i).value();
            const float yi = tensor::get_scalar(y, i).value();
            tensor::set_scalar(result, i, sigmoid_scalar(xi) * xi * yi);
        }
    }

}

export namespace spheres::builtin_qwen_embedding::transformer {

    constexpr std::size_t q_proj_l1 {2048};
    constexpr std::size_t q_proj_l2 {1024};
    constexpr std::size_t kv_proj_l1 {1024};
    constexpr std::size_t kv_proj_l2 {1024};
    constexpr std::size_t o_proj_l1 {1024};
    constexpr std::size_t o_proj_l2 {2048};
    constexpr std::size_t gate_proj_l1 {3072};
    constexpr std::size_t gate_proj_l2 {1024};
    constexpr std::size_t down_proj_l1 {1024};
    constexpr std::size_t down_proj_l2 {3072};

    void convert_bf16_to_float(const std::span<uint16_t> x, const tensor::Tensor &result) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            const auto ix = static_cast<uint32_t>(x[i]) << 16;
            tensor::set_scalar(result, i, std::bit_cast<float>(ix));
        }
    }

    // referenced https://github.com/adriancable/qwen3.c/blob/main/runq.c
    // and https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
    std::vector<float> calculate_qwen_embedding(const qwen_model::QwenWeights& weights, const std::string_view& query) {
        const std::vector<int> token_ids = tokenizer::encode(query);

        // setup buffers
        tensor::Tensor hidden_states {weights.config.hidden_size};
        // residual is also used for other buffers
        tensor::Tensor residual {weights.config.num_attention_heads * weights.config.head_dim};
        // [layer, seq_len, num_key_value_heads, head_dim]
        tensor::Tensor key_cache {weights.config.num_key_value_heads * weights.config.head_dim * weights.config.num_hidden_layers * token_ids.size()};
        tensor::Tensor value_cache {weights.config.num_key_value_heads * weights.config.head_dim * weights.config.num_hidden_layers * token_ids.size()};
        tensor::reshape(key_cache, weights.config.num_hidden_layers, token_ids.size(), weights.config.num_key_value_heads * weights.config.head_dim);
        tensor::reshape(value_cache, weights.config.num_hidden_layers, token_ids.size(), weights.config.num_key_value_heads * weights.config.head_dim);
        //
        tensor::Tensor q {weights.config.num_attention_heads * weights.config.head_dim};
        tensor::reshape(q, weights.config.num_attention_heads, weights.config.head_dim);
        tensor::Tensor attention {weights.config.num_attention_heads * token_ids.size()};
        tensor::reshape(attention, weights.config.num_attention_heads, token_ids.size());
        // mlp buffers
        tensor::Tensor mlp_buffer1 {weights.config.intermediate_size}; // first used in gate_proj
        tensor::Tensor mlp_buffer2 {weights.config.intermediate_size}; // first used in up_proj

        // forward loop
        for (const auto [pos, token_id] : std::views::enumerate(token_ids)) {
            convert_bf16_to_float(qwen_model::get_embedding_of_token(weights, token_id), hidden_states);
            for (std::size_t layer_index = 0; layer_index < weights.config.num_hidden_layers; ++layer_index) {
                tensor::Tensor k = key_cache[layer_index][pos];
                tensor::reshape(k, weights.config.num_key_value_heads, weights.config.head_dim);
                tensor::Tensor v = value_cache[layer_index][pos];
                auto input_layernorm_weight = qwen_model::get_weights_of_layer(weights, "input_layernorm_weight", layer_index, weights.config.hidden_size);
                math::rms_norm_float_bf16_weights(tensor::get_span_view(hidden_states), input_layernorm_weight, residual, weights.config.rms_norm_eps);

                auto self_attn_q_proj_weight =
                    array_view::ArrayView(qwen_model::get_weights_of_layer(weights, "self_attn_q_proj_weight", layer_index, q_proj_l1, q_proj_l2).data(), q_proj_l1, q_proj_l2);
                math::matrix_multiply_vector_bf16_f32(self_attn_q_proj_weight, tensor::get_span_view(residual), q);

                auto self_attn_k_proj_weight =
                    array_view::ArrayView(qwen_model::get_weights_of_layer(weights, "self_attn_k_proj_weight", layer_index, kv_proj_l1, kv_proj_l2).data(), kv_proj_l1, kv_proj_l2);
                math::matrix_multiply_vector_bf16_f32(self_attn_k_proj_weight, tensor::get_span_view(residual), k);

                auto self_attn_v_proj_weight =
                    array_view::ArrayView(qwen_model::get_weights_of_layer(weights, "self_attn_v_proj_weight", layer_index, kv_proj_l1, kv_proj_l2).data(), kv_proj_l1, kv_proj_l2);
                math::matrix_multiply_vector_bf16_f32(self_attn_v_proj_weight, tensor::get_span_view(residual), v);

                for (std::size_t head_index = 0; head_index < weights.config.num_attention_heads; ++head_index) {
                    tensor::Tensor current_q = q[head_index];
                    auto self_attn_q_norm_weight = qwen_model::get_weights_of_layer(weights, "self_attn_q_norm_weight", layer_index, weights.config.head_dim);
                    math::rms_norm_float_bf16_weights(tensor::get_span_view(current_q), self_attn_q_norm_weight, current_q, weights.config.rms_norm_eps);
                    for (std::size_t ri = 0; ri < weights.config.head_dim / 2; ++ri) {
                        const float frequency = powf(static_cast<float>(weights.config.rope_theta),
                            -static_cast<float>(ri) * 2 / static_cast<float>(weights.config.head_dim));
                        const float cos_frequency = cosf(frequency * static_cast<float>(pos));
                        const float sin_frequency = sinf(frequency * static_cast<float>(pos));
                        const float x = tensor::get_scalar(current_q, ri).value();
                        const float y = tensor::get_scalar(current_q, ri + weights.config.head_dim / 2).value();
                        tensor::set_scalar(current_q, ri, x * cos_frequency - y * sin_frequency);
                        tensor::set_scalar(current_q, ri + weights.config.head_dim / 2, x * sin_frequency + y * cos_frequency);
                    }
                }

                for (std::size_t head_index = 0; head_index < weights.config.num_key_value_heads; ++head_index) {
                    tensor::Tensor current_k = k[head_index];
                    auto self_attn_k_norm_weight = qwen_model::get_weights_of_layer(weights, "self_attn_k_norm_weight", layer_index, weights.config.head_dim);
                    math::rms_norm_float_bf16_weights(tensor::get_span_view(current_k), self_attn_k_norm_weight, current_k, weights.config.rms_norm_eps);
                    for (std::size_t ri = 0; ri < weights.config.head_dim / 2; ++ri) {
                        const float frequency = powf(static_cast<float>(weights.config.rope_theta),
                            -static_cast<float>(ri) * 2 / static_cast<float>(weights.config.head_dim));
                        const float cos_frequency = cosf(frequency * static_cast<float>(pos));
                        const float sin_frequency = sinf(frequency * static_cast<float>(pos));
                        const float x = tensor::get_scalar(current_k, ri).value();
                        const float y = tensor::get_scalar(current_k, ri + weights.config.head_dim / 2).value();
                        tensor::set_scalar(current_k, ri, x * cos_frequency - y * sin_frequency);
                        tensor::set_scalar(current_k, ri + weights.config.head_dim / 2, x * sin_frequency + y * cos_frequency);
                    }
                }

                const float sqrt_dim = sqrtf(static_cast<float>(weights.config.head_dim));
                for (std::size_t head_index = 0; head_index < weights.config.num_attention_heads; ++head_index) {
                    tensor::Tensor current_q = q[head_index];
                    tensor::Tensor current_attention = attention[head_index];
                    for (std::size_t p = 0; p <= pos; p++) {
                        tensor::Tensor k_at_pos_p = key_cache[layer_index][p];
                        tensor::reshape(k_at_pos_p, weights.config.num_key_value_heads, weights.config.head_dim);
                        tensor::Tensor current_k = k_at_pos_p[head_index / 2]; // 2 kv heads for 1 attention head
                        const float score = math::dot_product_float_float_vector(tensor::get_span_view(current_q), tensor::get_span_view(current_k));
                        tensor::set_scalar(current_attention, p, score / sqrt_dim);
                    }
                    tensor::Tensor current_attention_up_to_pos {static_cast<std::size_t>(pos + 1), current_attention.data, 1, static_cast<std::size_t>(pos + 1)};
                    math::softmax_vector_float(current_attention_up_to_pos);
                    tensor::reshape(residual, weights.config.num_attention_heads, weights.config.head_dim);
                    tensor::Tensor current_residual = residual[head_index];
                    tensor::flatten(residual);

                    {
                        tensor::Tensor v_at_pos_p = value_cache[layer_index][0];
                        tensor::reshape(v_at_pos_p, weights.config.num_key_value_heads, weights.config.head_dim);
                        tensor::Tensor current_v = v_at_pos_p[head_index / 2];
                        math::element_wise_fused_scalar_multiply_add_float_vector_overwrite_dst(
                            current_v, tensor::get_scalar(current_attention_up_to_pos, 0).value(), current_residual);
                    }
                    for (std::size_t p = 1; p <= pos; p++) {
                        tensor::Tensor v_at_pos_p = value_cache[layer_index][pos];
                        tensor::reshape(v_at_pos_p, weights.config.num_key_value_heads, weights.config.head_dim);
                        tensor::Tensor current_v = v_at_pos_p[head_index / 2];
                        math::element_wise_fused_scalar_multiply_add_float(
                            current_v, tensor::get_scalar(current_attention_up_to_pos, p).value(), current_residual);
                    }
                }
                auto self_attn_o_proj_weight =
                    array_view::ArrayView(qwen_model::get_weights_of_layer(weights, "self_attn_o_proj_weight", layer_index, o_proj_l1, o_proj_l2).data(), o_proj_l1, o_proj_l2);
                math::matrix_multiply_vector_bf16_f32(self_attn_o_proj_weight, tensor::get_span_view(residual), residual);
                for (std::size_t i = 0; i < weights.config.hidden_size; i++) {
                    tensor::set_scalar(hidden_states, i, tensor::get_scalar(hidden_states, i).value() + tensor::get_scalar(residual, i).value());
                }

                // mlp
                auto post_attention_layernorm_weight =
                    qwen_model::get_weights_of_layer(weights, "post_attention_layernorm_weight", layer_index, weights.config.hidden_size);
                math::rms_norm_float_bf16_weights(tensor::get_span_view(hidden_states), post_attention_layernorm_weight, residual, weights.config.rms_norm_eps);
                auto mlp_gate_proj_weight =
                    array_view::ArrayView(qwen_model::get_weights_of_layer(weights, "mlp_gate_proj_weight", layer_index, gate_proj_l1, gate_proj_l2).data(), gate_proj_l1, gate_proj_l2);
                math::matrix_multiply_vector_bf16_f32(mlp_gate_proj_weight, tensor::get_span_view(residual), mlp_buffer1);
                auto mlp_up_proj_weight =
                    array_view::ArrayView(qwen_model::get_weights_of_layer(weights, "mlp_up_proj_weight", layer_index, gate_proj_l1, gate_proj_l2).data(), gate_proj_l1, gate_proj_l2);
                math::matrix_multiply_vector_bf16_f32(mlp_up_proj_weight, tensor::get_span_view(residual), mlp_buffer2);
                math::swiglu_float_vector(mlp_buffer1, mlp_buffer2, mlp_buffer1);
                auto mlp_down_proj_weight =
                    array_view::ArrayView(qwen_model::get_weights_of_layer(weights, "mlp_down_proj_weight", layer_index, down_proj_l1, down_proj_l1).data(), down_proj_l1, down_proj_l2);
                math::matrix_multiply_vector_bf16_f32(mlp_down_proj_weight, tensor::get_span_view(mlp_buffer1), residual);
                for (std::size_t i = 0; i < weights.config.hidden_size; i++) {
                    tensor::set_scalar(hidden_states, i, tensor::get_scalar(hidden_states, i).value() + tensor::get_scalar(residual, i).value());
                }
            }

            // finally
            auto norm_weight = qwen_model::get_norm_weights(weights);
            math::rms_norm_float_bf16_weights(tensor::get_span_view(hidden_states), norm_weight, hidden_states, weights.config.rms_norm_eps);
        }
        std::vector<float> result_embedding;
        result_embedding.reserve(weights.config.hidden_size);
        std::ranges::copy_n(hidden_states.data, static_cast<long>(weights.config.hidden_size), std::back_inserter(result_embedding));

        return std::move(result_embedding);
    }

}