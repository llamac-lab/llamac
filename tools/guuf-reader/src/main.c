//
// Created by ervin on 7/11/25.
//
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

#include "gguf_reader.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    cursor_t cur;
    const int fd = cursor_open(argv[1], &cur);
    if (fd < 0) {
        perror("cursor_open failed");
        return 1;
    }

    char magic[5] = {0};
    for (int i = 0; i < 4; i++) {
        magic[i] = (char) cursor_read_u8(&cur);
    }

    if (strncmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Invalid GGUF magic: %.4s\n", magic);
        return 1;
    }

    const uint32_t version = cursor_read_u32_le(&cur);
    const uint64_t tensor_count = cursor_read_u64_le(&cur);
    const uint64_t kv_count = cursor_read_u64_le(&cur);

    printf("GGUF v%d | %llu tensors | %llu metadata kvs\n",
        version,
        (unsigned long long)tensor_count,
        (unsigned long long)kv_count
        );

    for (uint64_t i = 0; i < kv_count; ++i) {
        char *key = cursor_read_string(&cur);
        if (!key) break;

        value_t val = cursor_read_value(&cur);
        //printf("META[%llu] key = %s\n", i, key);
        if (val.type == META_ARRAY) {
            continue;
        }

        printf("[%s] = ", key);
        switch (val.type) {
            case META_STRING: {
                printf(" string = %s\n", val.string);
                break;
            }
            // case META_ARRAY: {
            //     printf("array = []\n");
            //     break;
            // }
            case META_U8: {
                printf(" u8 = %u\n", val.u8);
                break;
            }
            case META_I8: {
                printf(" i8 = (int8_t)%u\n", val.i8);
                break;
            }
            case META_U16: {
                printf(" u16 = %u\n", val.u16);
                break;
            }
            case META_I16: {
                printf(" i16 = (int16_t)%u\n", val.i16);
                break;
            }
            case META_U32: {
                printf(" u32 = %u\n", val.u32);
                break;
            }
            case META_I32: {
                printf(" i32 = (int32_t)%u\n", val.i32);
                break;
            }
            case META_U64: {
                printf(" u64 = %lu\n", val.u64);
                break;
            }
            case META_I64: {
                printf(" i64 = (int64_t)%ld\n", val.i64);
                break;
            }
            case META_F32: {
                printf(" f32 = %f\n", val.f32);
                break;
            }
            case META_F64: {
                printf(" f64 = %f\n", val.f64);
                break;
            }
            case META_BOOL: {
                printf(" boolean = %u\n", val.boolean);
                break;
            }
            default: {
                printf("unknown metadata type tag: %u\n", val.type);
                break;
            }

        }

        // if (val.type == META_STRING) {
        //     printf(" string = %s\n", val.string);
        // }
        // if (val.type == META_STRING) {
        //     printf(" string = %s\n", val.string);
        // }

        free(key);
        value_free(&val);
    }
    cursor_close(&cur);
    close(fd);
    return 0;
}