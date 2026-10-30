#ifndef DUCKTAPE_H
#define DUCKTAPE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define DUCKTAPE_ABI_VERSION 1

typedef struct ducktape_Operation ducktape_Operation;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Returns the ABI version.
 */
uint32_t ducktape_abi_version(void);

/**
 * Allocates `len` bytes the host may write into; free with `ducktape_free`.
 */
uint8_t *ducktape_alloc(size_t len);

/**
 * Frees a buffer of `len` bytes that came from `ducktape_alloc` or an out parameter.
 */
void ducktape_free(uint8_t *ptr, size_t len);

/**
 * Starts an operation from a `Start` document and writes the first `Step`; null handle on an invalid document.
 */
ducktape_Operation *ducktape_op_start(const uint8_t *json,
                                      size_t len,
                                      uint8_t **out,
                                      size_t *out_len);

/**
 * Resumes an operation with a `Resume` document and writes the next `Step`.
 */
int32_t ducktape_op_resume(ducktape_Operation *op,
                           const uint8_t *json,
                           size_t len,
                           uint8_t **out,
                           size_t *out_len);

/**
 * Delivers raw body bytes of the open response with a `Clock` document and writes the next `Step`.
 */
int32_t ducktape_op_feed(ducktape_Operation *op,
                         const uint8_t *bytes,
                         size_t bytes_len,
                         const uint8_t *clock_json,
                         size_t clock_len,
                         uint8_t **out,
                         size_t *out_len);

/**
 * Stores a secret under `secret_ref`; the core copies the bytes and the caller wipes its own.
 */
int32_t ducktape_op_set_secret(ducktape_Operation *op,
                               const uint8_t *secret_ref,
                               size_t ref_len,
                               const uint8_t *bytes,
                               size_t len);

/**
 * Frees an operation handle.
 */
void ducktape_op_free(ducktape_Operation *op);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* DUCKTAPE_H */
