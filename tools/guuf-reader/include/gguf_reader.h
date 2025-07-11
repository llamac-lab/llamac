//
// Created by ervin on 7/11/25.
//

#ifndef GGUF_READER_H
#define GGUF_READER_H

#pragma once
#include <stdint.h>
#include <stddef.h>

typedef enum {
    META_U8      = 0,
    META_I8      = 1,
    META_U16     = 2,
    META_I16     = 3,
    META_U32     = 4,
    META_I32     = 5,
    META_F32     = 6,
    META_BOOL    = 7,
    META_STRING  = 8,
    META_ARRAY   = 9,
    META_U64     = 10,
    META_I64     = 11,
    META_F64     = 12
} meta_type_t;

typedef struct value_t {
    meta_type_t type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        double   f64;
        uint64_t u64;
        int64_t  i64;
        uint8_t  boolean;

        char *string;

        struct {
            meta_type_t subtype;
            size_t len;
            struct value_t *items;
        } array;
    };
} value_t;

typedef struct {
    char *key;
    value_t value;
} kv_pair_t;

typedef struct {
    char *name;
    uint32_t type;
    uint32_t ndim;
    uint32_t shape[4];
    uint32_t offset;
    uint32_t size;
} tensor_t;

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t offset;
} cursor_t;

int         cursor_open(const char *filepath, cursor_t *out);
void        cursor_close(cursor_t *cursor);
uint8_t     cursor_read_u8(cursor_t *cur);
uint32_t    cursor_read_u32_le(cursor_t *cur);
uint64_t    cursor_read_u64_le(cursor_t *cur);
char *      cursor_read_string(cursor_t *cur);
value_t     cursor_read_value(cursor_t *cur);
void        value_free(value_t *val);

//
int parse_metadata_common(cursor_t *cur, kv_pair_t **out, size_t *count);
int parse_tensors_common(cursor_t *cur, tensor_t **out, size_t *count);
char *cursor_read_string(cursor_t *c);

#endif //GGUF_READER_H
