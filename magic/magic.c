#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#define popcnt(x) __builtin_popcountll(x)
#define ffs(x) (__builtin_ffsll(x) - 1)

void print_bb(uint64_t b) {
    for (int rk=7; rk>=0; rk--) {
        for(int fl=0; fl<8; fl++) {
            int i=8*rk + fl;
            if (b & (1ULL << i)) {
                printf("x");
            } else {
                printf(".");
            }
        }
        printf("\n");
    }
}

void print_bb2(uint64_t a, uint64_t b) {
    for (int rk=7; rk>=0; rk--) {
        for(int fl=0; fl<8; fl++) {
            int i=8*rk + fl;
            if (a & (1ULL << i)) {
                printf("x");
            } else {
                printf(".");
            }
        }
        printf(" ");
        for(int fl=0; fl<8; fl++) {
            int i=8*rk + fl;
            if (b & (1ULL << i)) {
                printf("x");
            } else {
                printf(".");
            }
        }
        printf("\n");
    }
}

uint64_t rook_mask(int sq) {
    int rk = sq >> 3, fl = sq & 7;
    uint64_t mask = 0ULL;
    for (int i = rk+1; i < 7; i++) mask |= (1ULL << (i << 3 | fl));
    for (int i = rk-1; i > 0; i--) mask |= (1ULL << (i << 3 | fl));
    for (int i = fl+1; i < 7; i++) mask |= (1ULL << (rk << 3 | i));
    for (int i = fl-1; i > 0; i--) mask |= (1ULL << (rk << 3 | i));
    return mask;
}

uint64_t rook_attacks(int sq, uint64_t blockers) {
    int rk = sq >> 3, fl = sq & 7;
    uint64_t mask = 0ULL;
    for (int i = rk+1; i < 8; i++) {
        mask |= (1ULL << (i << 3 | fl));
        if (blockers & (1ULL << (i << 3 | fl))) break;
    }
    for (int i = rk-1; i >= 0; i--) {
        mask |= (1ULL << (i << 3 | fl));
        if (blockers & (1ULL << (i << 3 | fl))) break;
    }
    for (int i = fl+1; i < 8; i++) {
        mask |= (1ULL << (rk << 3 | i));
        if (blockers & (1ULL << (rk << 3 | i))) break;
    }
    for (int i = fl-1; i >= 0; i--) {
        mask |= (1ULL << (rk << 3 | i));
        if (blockers & (1ULL << (rk << 3 | i))) break;
    }
    return mask;
}

uint64_t bishop_mask(int sq) {
    int rk = sq >> 3, fl = sq & 7;
    uint64_t mask = 0ULL;
    for (int r=rk+1, f=fl+1; r<7 && f<7; r++, f++) mask |= (1ULL << (r << 3 | f));
    for (int r=rk-1, f=fl+1; r>0 && f<7; r--, f++) mask |= (1ULL << (r << 3 | f));
    for (int r=rk+1, f=fl-1; r<7 && f>0; r++, f--) mask |= (1ULL << (r << 3 | f));
    for (int r=rk-1, f=fl-1; r>0 && f>0; r--, f--) mask |= (1ULL << (r << 3 | f));
    return mask;
}

uint64_t bishop_attacks(int sq, uint64_t blockers) {
    int rk = sq >> 3, fl = sq & 7;
    uint64_t mask = 0ULL;
    for (int r=rk+1, f=fl+1; r<8 && f<8; r++, f++) {
        mask |= (1ULL << (r << 3 | f));
        if (blockers & (1ULL << (r << 3 | f))) break;
    }
    for (int r=rk-1, f=fl+1; r>=0 && f<8; r--, f++) {
        mask |= (1ULL << (r << 3 | f));
        if (blockers & (1ULL << (r << 3 | f))) break;
    }
    for (int r=rk+1, f=fl-1; r<8 && f>=0; r++, f--) {
        mask |= (1ULL << (r << 3 | f));
        if (blockers & (1ULL << (r << 3 | f))) break;
    }
    for (int r=rk-1, f=fl-1; r>=0 && f>=0; r--, f--) {
        mask |= (1ULL << (r << 3 | f));
        if (blockers & (1ULL << (r << 3 | f))) break;
    }
    return mask;
}

uint64_t clear_bits(uint64_t mask, int n) {
    uint64_t tmp = mask;
    for (int i=0; tmp; i++) {
        int bi = ffs(tmp);
        tmp &= ~(1ULL << bi);
        if ((n & (1 << i)) == 0) mask &= ~(1ULL << bi);
    }
    return mask;
}

int transform(uint64_t magic, uint64_t blockers, int bits) {
    return (unsigned)( (int)blockers * (int)magic ^ (int)(blockers>>32) * (int)(magic>>32) ) >> (32-bits);
}

bool test_magic(int sq, uint64_t magic, uint64_t mask) {
    int bits = popcnt(mask);
    uint64_t *buffer = calloc(1 << bits, sizeof(uint64_t));
    bool fail = 0;

    for (int i=0; !fail && i < 1<<bits; i++) {
        uint64_t blockers = clear_bits(mask, i);
        uint64_t attacks = rook_attacks(sq, blockers);
        int index = transform(magic, blockers, bits);
        if (buffer[index] == 0) {
            buffer[index] = attacks;
        } else {
            if (buffer[index] != attacks) fail = true;
        }
    }
    free(buffer);
    return !fail;
}

uint64_t rand64() {
    return (rand() & 0xffff) | (rand() & 0xffff) << 16
        | (uint64_t)(rand() & 0xffff) << 32 | (uint64_t)(rand() & 0xffff) << 48;
}

uint64_t rand64_few_bits() {
    return rand64() & rand64() & rand64() & rand64() & rand64();
}

int main(int argc, char *argv[]) {

    srand(time(NULL));

    for (int sq = 0; sq < 64; sq++) {
            uint64_t mask = rook_mask(sq);

            bool res = false;
            uint64_t magic;
            while (!res) {
                magic = rand64_few_bits();
                res = test_magic(sq, magic, mask);
            }
            printf("0x%llx %d\n", magic, popcnt(magic));
    }
}
