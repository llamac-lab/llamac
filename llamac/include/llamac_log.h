//
// Created by ervin on 7/13/25.
//

#pragma once
#ifdef __cplusplus
extern "C" {
#endif

    /* 0 = silent, 1 = errors, 2 = warn, 3 = info, 4 = debug */
    void llamac_set_log_level(int min_level);


#ifdef __cplusplus
}
#endif