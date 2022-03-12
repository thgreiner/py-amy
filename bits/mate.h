#ifndef MATE_H
#define MATE_H

#ifdef __cplusplus
extern "C" {
#endif

void test_mate_search();
uint32_t mate_search(heap_t, const position_t, int, uint64_t);

#ifdef __cplusplus
}
#endif

#endif
