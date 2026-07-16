/*
 * stdendian.h - minimal byte-order shim for the vendored Fathom (tbprobe.c).
 *
 * Provides exactly what Fathom references: the _BYTE_ORDER / _LITTLE_ENDIAN /
 * _BIG_ENDIAN macros and bswap16/32/64. Correct for x86/x64 (little-endian),
 * which is this project's target. If you have Fathom's official stdendian.h,
 * that canonical file may be used instead.
 */
#ifndef STDENDIAN_H
#define STDENDIAN_H

#include <stdint.h>

#define _LITTLE_ENDIAN 1234
#define _BIG_ENDIAN    4321

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define _BYTE_ORDER _BIG_ENDIAN
#else
#  define _BYTE_ORDER _LITTLE_ENDIAN     /* x86 / x64 / ARM little-endian */
#endif

#if defined(_MSC_VER)
#  include <stdlib.h>
#  define bswap16(x) _byteswap_ushort((uint16_t)(x))
#  define bswap32(x) _byteswap_ulong((uint32_t)(x))
#  define bswap64(x) _byteswap_uint64((uint64_t)(x))
#elif defined(__GNUC__) || defined(__clang__)
#  define bswap16(x) __builtin_bswap16((uint16_t)(x))
#  define bswap32(x) __builtin_bswap32((uint32_t)(x))
#  define bswap64(x) __builtin_bswap64((uint64_t)(x))
#else
static inline uint16_t bswap16(uint16_t x) { return (uint16_t)((x >> 8) | (x << 8)); }
static inline uint32_t bswap32(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}
static inline uint64_t bswap64(uint64_t x) {
    return ((uint64_t)bswap32((uint32_t)x) << 32) | bswap32((uint32_t)(x >> 32));
}
#endif

#endif /* STDENDIAN_H */
