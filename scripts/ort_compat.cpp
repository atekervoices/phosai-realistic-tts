// Compatibility shim for prebuilt ONNX Runtime (ort-sys) binaries compiled against
// newer glibc (2.38+ C23 strto* symbols) and GCC 13 libstdc++ (_M_replace_cold)
// when linking on Ubuntu 22.04 (glibc 2.35 / GCC 11).

#include <cstdlib>
#include <cstring>
#include <cwchar>

extern "C" {

long __isoc23_strtol(const char* nptr, char** endptr, int base) {
    return std::strtol(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char* nptr, char** endptr, int base) {
    return std::strtoul(nptr, endptr, base);
}

long long __isoc23_strtoll(const char* nptr, char** endptr, int base) {
    return std::strtoll(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char* nptr, char** endptr, int base) {
    return std::strtoull(nptr, endptr, base);
}

} // extern "C"

namespace std {
template <class _CharT>
struct char_traits;
template <typename _Tp>
class allocator;

namespace __cxx11 {
template <typename _CharT, typename _Traits = char_traits<_CharT>, typename _Alloc = allocator<_CharT>>
class basic_string {
public:
    void _M_replace_cold(_CharT* __p, unsigned long __len1, const _CharT* __s,
                         const unsigned long __len2, const unsigned long __how_much);
};

template <typename _CharT>
static inline void replace_cold_impl(_CharT* __p, unsigned long __len1, const _CharT* __s,
                                     const unsigned long __len2, const unsigned long __how_much) {
    if (__how_much && __len1 != __len2) {
        std::memmove(__p + __len2, __p + __len1, __how_much * sizeof(_CharT));
    }
    if (__s + __len2 <= __p || __s >= __p + __len1) {
        std::memcpy(__p, __s, __len2 * sizeof(_CharT));
    } else if (__s < __p) {
        const unsigned long __nleft = static_cast<unsigned long>(__p - __s);
        std::memmove(__p, __s, __nleft * sizeof(_CharT));
        std::memcpy(__p + __nleft, __p + __len2, (__len2 - __nleft) * sizeof(_CharT));
    } else {
        const unsigned long __nright = static_cast<unsigned long>((__s + __len2) - (__p + __len1));
        std::memmove(__p + __len2 - __nright, __s + (__len2 - __nright), __nright * sizeof(_CharT));
        std::memmove(__p, __p + __len2, (__len2 - __nright) * sizeof(_CharT));
    }
}

template <>
void basic_string<char, char_traits<char>, allocator<char>>::_M_replace_cold(
    char* __p, unsigned long __len1, const char* __s,
    const unsigned long __len2, const unsigned long __how_much) {
    replace_cold_impl(__p, __len1, __s, __len2, __how_much);
}

template <>
void basic_string<wchar_t, char_traits<wchar_t>, allocator<wchar_t>>::_M_replace_cold(
    wchar_t* __p, unsigned long __len1, const wchar_t* __s,
    const unsigned long __len2, const unsigned long __how_much) {
    replace_cold_impl(__p, __len1, __s, __len2, __how_much);
}

} // namespace __cxx11
} // namespace std

