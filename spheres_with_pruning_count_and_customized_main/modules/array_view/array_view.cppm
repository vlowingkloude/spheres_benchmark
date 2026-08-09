module;

#include <span>
#include <cstddef>
#include <stdexcept>

export module array_view;

export namespace spheres::array_view {

    template <typename T>
    struct ArrayView {
        T* data;
        std::size_t s1;
        std::size_t s2;
        std::size_t size;

        ArrayView() {
            data = nullptr;
            s1 = s2 = 0;
            size = 0;
        }

        explicit ArrayView(T *data, const std::size_t size) : data(data), size(size) {
            s1 = 0;
            s2 = 0;
        }

        ArrayView(T *data, const std::size_t s1, const std::size_t s2) : data(data), s1(s1), s2(s2) {
            size = s1 * s2;
        }

        ArrayView (const ArrayView& other) {
            data = other.data;
            s1 = other.s1;
            s2 = other.s2;
            size = other.size;
        }
        ArrayView& operator=(const ArrayView& other) {
            if (this == &other) {
                return *this;
            }
            data = other.data;
            s1 = other.s1;
            s2 = other.s2;
            size = other.size;
            return *this;
        }

        std::span<T> operator[](const std::size_t index) {
            if (index >= s1 || data == nullptr) {
                return {};
            } else {
                return std::span{data + index * s2, s2};
            }
        }

        ArrayView (ArrayView&& other) noexcept : data(other.data), s1(other.s1), s2(other.s2), size(other.size) {
            other.data = nullptr;
            other.size = 0;
            other.s1 = 0;
            other.s2 = 0;
        }

        ~ArrayView() {
            data = nullptr;
            s1 = s2 = 0;
            size = 0;
        }
    };

    template <typename T>
    void reshape_array(ArrayView<T>& array, std::size_t s1, std::size_t s2) {
        if (s1 * s2 != array.size) {
            throw std::out_of_range ("shape does not match array size");
        }
        array.s1 = s1;
        array.s2 = s2;
    }

}