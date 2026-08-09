module;

#include <cstdint>
#include <string_view>
#include <vector>
#include <string>

export module builtin_qwen_tokenizer;

namespace spheres::builtin_qwen_embedding::tokenizer::generated {
    int find_in_merge(unsigned long s);
    int get_id_for_token(unsigned long token);
}

namespace spheres::builtin_qwen_embedding::tokenizer::constants {

    static constexpr signed char UTF8_LOOKUP_TABLE[] = {
        0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,2,2,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,
        3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,-1,-1,-1,-1,-1,-1,-1,-1
    };

    static const unsigned short BYTE_TO_UNICODE_LOOKUP_TABLE[] = {
        32964, 33220, 33476, 33732, 33988, 34244, 34500, 34756, 35012, 35268, 35524,
        35780, 36036, 36292, 36548, 36804, 37060, 37316, 37572, 37828, 38084,
        38340, 38596, 38852, 39108, 39364, 39620, 39876, 40132, 40388, 40644,
        40900, 41156, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        121, 122, 123, 124, 125, 126, 41412, 41668, 41924, 42180,
        42436, 42692, 42948, 43204, 43460, 43716, 43972, 44228, 44484, 44740,
        44996, 45252, 45508, 45764, 46020, 46276, 46532, 46788, 47044, 47300,
        47556, 47812, 48068, 48324, 48580, 48836, 49092, 32965, 33221, 33477,
        41410, 41666, 41922, 42178, 42434, 42690, 42946, 43202, 43458, 43714,
        43970, 44226, 33733, 44738, 44994, 45250, 45506, 45762, 46018, 46274,
        46530, 46786, 47042, 47298, 47554, 47810, 48066, 48322, 48578, 48834,
        49090, 32963, 33219, 33475, 33731, 33987, 34243, 34499, 34755, 35011,
        35267, 35523, 35779, 36035, 36291, 36547, 36803, 37059, 37315, 37571,
        37827, 38083, 38339, 38595, 38851, 39107, 39363, 39619, 39875, 40131,
        40387, 40643, 40899, 41155, 41411, 41667, 41923, 42179, 42435, 42691,
        42947, 43203, 43459, 43715, 43971, 44227, 44483, 44739, 44995, 45251,
        45507, 45763, 46019, 46275, 46531, 46787, 47043, 47299, 47555, 47811,
        48067, 48323, 48579, 48835, 49091
    };

}

namespace spheres::builtin_qwen_embedding::tokenizer {

    unsigned short convert_byte_to_unicode(const uint8_t byte) {
        return constants::BYTE_TO_UNICODE_LOOKUP_TABLE[byte];
    }

    unsigned char convert_unicode_to_byte(const uint16_t unicode) {
        if (unicode < 128) {
            return unicode;
        }
        for (int i = 0; i < 256; i++) {
            if (constants::BYTE_TO_UNICODE_LOOKUP_TABLE[i] == unicode) {
                return i;
            }
        }
        return 0;
    }

    int use_n_bytes(const unsigned char c) {
        return constants::UTF8_LOOKUP_TABLE[c];
    }

}

namespace spheres::builtin_qwen_embedding::tokenizer::generated {
    bool have_conflicts_merge();

    bool have_conflicts_token();
}

export namespace spheres::builtin_qwen_embedding::tokenizer {

    std::vector<int> encode(const std::string_view& text) {
        if (text.empty()) {
            return {};
        }
        std::string replaced_text;
        replaced_text.reserve(text.size() * 2);
        for (const unsigned char t : text) {
            if (const unsigned short code = convert_byte_to_unicode(t); code < 128) {
                replaced_text.push_back(static_cast<char>(code));
            } else {
                replaced_text.push_back(static_cast<char>(code & 0x00ff));
                replaced_text.push_back(static_cast<char>((code >> 8) & 0x00ff));
            }
        }
        std::vector<std::string_view> tokens;
        tokens.reserve(replaced_text.size());
        for (std::size_t i = 0; i < replaced_text.size();) {
            auto size = static_cast<std::size_t>(use_n_bytes(static_cast<unsigned char>(replaced_text[i])));
            tokens.emplace_back(replaced_text.data() + i, size);
            i += size;
        }

        bool changed = true;
        while (changed) {
            changed = false;
            int best_rank = -1;
            std::size_t best_ls = std::string_view::npos;
            std::size_t best_rs = std::string_view::npos;
            for (std::size_t i = 0; i < tokens.size();) {
                const std::string_view ls { tokens[i] };
                std::size_t k = i + 1;
                while (k < tokens.size() && tokens[k].empty()) {
                    k++;
                }
                if (k >= tokens.size()) {
                    break;
                }
                const std::string_view rs { tokens[k] };
                const std::string_view token { ls.data(), ls.size() + rs.size() };
                if (const int rank = generated::find_in_merge(std::hash<std::string_view>{}(token)); rank != -1) {
                    if (best_rank == -1 || rank < best_rank) {
                        best_rank = rank;
                        best_ls = i;
                        best_rs = k;
                    }
                }
                i = k;
            }
            if (best_rank != -1) {
                changed = true;
                std::string_view& left = tokens[best_ls];
                std::string_view& right = tokens[best_rs];
                left = std::string_view(left.data(), left.size() + right.size());
                right = {};
            }
        }
        std::erase_if(tokens, [](const std::string_view& t) { return t.empty(); });
        std::vector<int> ids;
        ids.reserve(tokens.size());
        for (const auto& t : tokens) {
            ids.emplace_back(generated::get_id_for_token(std::hash<std::string_view>{}(t)));
        }
        return ids;
    }

}