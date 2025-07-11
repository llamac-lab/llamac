#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

#include "gguf_reader.h"

int cursor_open(const char *filepath, cursor_t *out) {

    const int fd = open(filepath, O_RDONLY);
    if (fd < 0) { return -1; }

    struct stat st;
    if (fstat(fd, &st) <0 ) {
        close(fd);
        return -1;
    }

    void *mmaped = mmap( NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmaped == MAP_FAILED) {
        close(fd);
        return -1;
    }

    out->data = (const uint8_t *)mmaped;
    out->size = st.st_size;
    out->offset = 0;

    return 0;
}

void cursor_close(cursor_t *cursor) {
    if (cursor->data) {
        munmap((void *)cursor->data, cursor->size);
        cursor->data = NULL;
        cursor->size = 0;
        cursor->offset = 0;
    }
}

// basic endian functions
uint8_t cursor_read_u8(cursor_t *cur) {
    return cur->data[cur->offset++];
}

uint32_t cursor_read_u32_le(cursor_t *c) {
    const uint8_t *p = &c->data[c->offset];
    c->offset += 4;
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

uint64_t cursor_read_u64_le(cursor_t *c) {
    const uint8_t *p = &c->data[c->offset];
    c->offset += 8;
    return (uint64_t)p[0] |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

char *cursor_read_string(cursor_t *c) {
    const uint64_t len = cursor_read_u64_le(c);
    if (c->offset + len > c->size) {
        fprintf(stderr, "String read exceeds buffer size\n");
        return NULL;
    }

    char *s = malloc(len + 1);
    if (!s) return NULL;

    memcpy(s, c->data + c->offset, len);
    s[len] = 0;
    c->offset += len;
    return s;
}

value_t cursor_read_value(cursor_t *c) {
    value_t val = {0};
    uint32_t tag = cursor_read_u32_le(c);
    val.type = (meta_type_t)tag;

    switch (val.type) {
        case META_U8:    val.u8  = cursor_read_u8(c);  break;
        case META_I8:    val.i8  = (int8_t)cursor_read_u8(c); break;
        case META_U16:   val.u16 = cursor_read_u32_le(c); break;
        case META_I16:   val.i16 = (int16_t)cursor_read_u32_le(c); break;
        case META_U32:   val.u32 = cursor_read_u32_le(c); break;
        case META_I32:   val.i32 = (int32_t)cursor_read_u32_le(c); break;
        case META_U64:   val.u64 = cursor_read_u64_le(c); break;
        case META_I64:   val.i64 = (int64_t)cursor_read_u64_le(c); break;
        case META_F32: {
            uint32_t raw = cursor_read_u32_le(c);
            val.f32 = *(float *)&raw;
            break;
        }
        case META_F64: {
            uint64_t raw = cursor_read_u64_le(c);
            val.f64 = *(double *)&raw;
            break;
        }
        case META_BOOL:  val.boolean = cursor_read_u8(c) ? 1 : 0; break;

        case META_STRING: {
            val.string = cursor_read_string(c);
            break;
        }

        case META_ARRAY: {
            meta_type_t subtype = (meta_type_t)cursor_read_u32_le(c);
            uint64_t count = cursor_read_u64_le(c);

            val.array.subtype = subtype;
            val.array.len = count;
            val.array.items = calloc(count, sizeof(value_t));
            for (size_t i = 0; i < count; ++i) {
                // recurse manually for subtype
                val.array.items[i].type = subtype;
                switch (subtype) {
                    case META_U8:    val.array.items[i].u8  = cursor_read_u8(c); break;
                    case META_I8:    val.array.items[i].i8  = (int8_t)cursor_read_u8(c); break;
                    case META_U16:   val.array.items[i].u16 = cursor_read_u32_le(c); break;
                    case META_I16:   val.array.items[i].i16 = (int16_t)cursor_read_u32_le(c); break;
                    case META_U32:   val.array.items[i].u32 = cursor_read_u32_le(c); break;
                    case META_I32:   val.array.items[i].i32 = (int32_t)cursor_read_u32_le(c); break;
                    case META_U64:   val.array.items[i].u64 = cursor_read_u64_le(c); break;
                    case META_I64:   val.array.items[i].i64 = (int64_t)cursor_read_u64_le(c); break;
                    case META_F32: {
                        uint32_t raw = cursor_read_u32_le(c);
                        val.array.items[i].f32 = *(float *)&raw;
                        break;
                    }
                    case META_F64: {
                        uint64_t raw = cursor_read_u64_le(c);
                        val.array.items[i].f64 = *(double *)&raw;
                        break;
                    }
                    case META_BOOL: val.array.items[i].boolean = cursor_read_u8(c); break;
                    case META_STRING: val.array.items[i].string = cursor_read_string(c); break;
                    default: fprintf(stderr, "Unsupported nested array subtype %d\n", subtype);
                             break;
                }
            }
            break;
        }

        default:
            fprintf(stderr, "Unknown metadata type tag: %u\n", tag);
            break;
    }

    return val;
}

void value_free(value_t *val) {
    if (!val) return;

    if (val->type == META_STRING && val->string) {
        free(val->string);
    } else if (val->type == META_ARRAY) {
        for (size_t i = 0; i < val->array.len; ++i)
            value_free(&val->array.items[i]);
        free(val->array.items);
    }
}//
// Created by ervin on 7/11/25.
//
