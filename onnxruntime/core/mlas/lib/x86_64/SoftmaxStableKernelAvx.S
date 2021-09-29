/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SoftmaxStableKernelAvx.s

Abstract:

    This module implements the kernels for the single precision softmax
    operation.

    This implementation uses AVX instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

// Constants
        .equ    .LExpConstants_LowerRange, 0
        .equ    .LExpConstants_UpperRange, 4
        .equ    .LExpConstants_LowerRangeSumExp, 8
        .equ    .LExpConstants_UpperRangeSumExp, 12
        .equ    .LExpConstants_RoundingBias, 16
        .equ    .LExpConstants_Log2Reciprocal, 20
        .equ    .LExpConstants_Log2High, 24
        .equ    .LExpConstants_Log2Low, 28
        .equ    .LExpConstants_poly_0, 32
        .equ    .LExpConstants_poly_1, 36
        .equ    .LExpConstants_poly_2, 40
        .equ    .LExpConstants_poly_3, 44
        .equ    .LExpConstants_poly_4, 48
        .equ    .LExpConstants_poly_56, 52
        .equ    .LExpConstants_MinimumExponent, 56
        .equ    .LExpConstants_MaximumExponent, 60

/*++

Routine Description:

    Softmax

Arguments:

    Input (rdi) - Supplies the input buffer.

    Output (rsi) - Supplies the output buffer.

    N (rdx) - Supplies the number of elements to process.

    rcx - moving Input
    r8  - moving Output
    r9  - moving N
    r10 - MlasOpmask16BitTableAvx512

    rax - tmp: MlasExpConstants, for k1 mask

Return Value:

    Softmax

--*/

         .globl  C_UNDERSCORE(MlasSoftmaxStableAvx)
C_UNDERSCORE(MlasSoftmaxStableAvx):

/*=============== Loop1: max ==============*/

        # moving Input and N, Output is not used in this loop
        mov rcx,rdi
        mov r9,rdx

        vbroadcastss ymm0,DWORD PTR C_UNDERSCORE(MlasMinimumF32Value)[rip]

        test    r9,r9                                         # N == 0, just exit
        jz      .LSoftmaxStable.ExitReduceMax
        cmp     r9,8
        jb      .LSoftmaxStable.ProcessReduceMaxBy1
        cmp     r9,32
        jb      .LSoftmaxStable.ProcessReduceMaxBy8

        vmovaps ymm1,ymm0                       # expand min to ymm0 1 2 3
        vmovaps ymm2,ymm0
        vmovaps ymm3,ymm0

.LSoftmaxStable.ProcessReduceMaxBy4x8:
        vmaxps  ymm0,ymm0,YMMWORD PTR [rcx]
        vmaxps  ymm1,ymm1,YMMWORD PTR [rcx+8*4]
        sub     r9,32                          # less data dep
        vmaxps  ymm2,ymm2,YMMWORD PTR [rcx+16*4]
        vmaxps  ymm3,ymm3,YMMWORD PTR [rcx+24*4]
        add     rcx,32*4                        # advance input by 32 elements
        cmp     r9,32
        jae     .LSoftmaxStable.ProcessReduceMaxBy4x8
        vmaxps  ymm0,ymm0,ymm1                  # reduce to single vector
        vmaxps  ymm2,ymm2,ymm3
        vmaxps  ymm0,ymm0,ymm2

.LSoftmaxStable.ProcessReduceMaxBy8:
        cmp     r9,8
        jb      .LSoftmaxStable.ProcessReduceMaxByLT8
        vmaxps  ymm0,ymm0,YMMWORD PTR [rcx]
        sub     r9,8
        add     rcx,8*4                         # advance input by 8 elements
        jmp     .LSoftmaxStable.ProcessReduceMaxBy8

.LSoftmaxStable.ProcessReduceMaxByLT8: # reduce to single value
        vextractf128 xmm1,ymm0,1                # 4 in xmm0, 4 in xmm1
        vmaxps  xmm0,xmm0,xmm1                  # reduce to xmm0
        vshufps xmm1,xmm0,xmm0,0xEE             # 2323
        vmaxps  xmm0,xmm0,xmm1                  # shrink to 2 elements
        vshufps xmm1,xmm0,xmm0,0x55             # 1111
        vmaxss  xmm0,xmm0,xmm1                  # first element
        test    r9,r9
        jz      .LSoftmaxStable.ExitReduceMax

.LSoftmaxStable.ProcessReduceMaxBy1:       # one by one
        // TODO: use k1
        vmaxss  xmm0,xmm0,DWORD PTR [rcx]
        add     rcx,4                           # advance input by 1 element
        dec     r9d                             # N-=1
        jnz     .LSoftmaxStable.ProcessReduceMaxBy1

/*=============== Loop2: exp & sum ==============*/

.LSoftmaxStable.ExitReduceMax:

        # moving Input Output and N
        mov rcx,rdi
        mov r8,rsi
        mov r9,rdx

        lea     rax,C_UNDERSCORE(MlasExpConstants)[rip]
        vbroadcastss zmm21,.LExpConstants_LowerRange[rax]
        vbroadcastss zmm22,.LExpConstants_RoundingBias[rax]
        vbroadcastss zmm23,.LExpConstants_Log2Reciprocal[rax]
        vbroadcastss zmm24,.LExpConstants_Log2High[rax]
        vbroadcastss zmm25,.LExpConstants_Log2Low[rax]
        vbroadcastss zmm26,.LExpConstants_poly_0[rax]
        vbroadcastss zmm27,.LExpConstants_poly_1[rax]
        vbroadcastss zmm28,.LExpConstants_poly_2[rax]
        vbroadcastss zmm29,.LExpConstants_poly_3[rax]
        vbroadcastss zmm30,.LExpConstants_poly_4[rax]
        vbroadcastss zmm31,.LExpConstants_poly_56[rax]

        vpxord  zmm20,zmm20,zmm20               # clear exp() accumulator
        vsubss  xmm0,xmm20,xmm0      # tage negation of max
        vbroadcastss zmm19,xmm0      # broadcast negative maximum value
        sub     r9,48                          # 3 vector a time
        jb      .LSoftmaxStable.ComputeExpByLT3x16

.LSoftmaxStable.ComputeExpBy3x16:
        vaddps  zmm0,zmm19,ZMMWORD PTR [rcx]    # bias by negative maximum value, higher precision
        vaddps  zmm3,zmm19,ZMMWORD PTR [rcx+64]
        vaddps  zmm16,zmm19,ZMMWORD PTR [rcx+128]

        # clamp 0 3 16, 1/LN2 to 2 5 18
        vmaxps  zmm0,zmm21,zmm0                 # clamp lower bound
        vmovaps zmm2,zmm23
        vmaxps  zmm3,zmm21,zmm3
        vmovaps zmm5,zmm23
        vmaxps  zmm16,zmm21,zmm16
        vmovaps zmm18,zmm23

        vfmadd213ps zmm2,zmm0,zmm22             # (input / ln2) plus rounding bias
        vfmadd213ps zmm5,zmm3,zmm22
        vfmadd213ps zmm18,zmm16,zmm22

        vmovaps zmm1,zmm26                      # P0
        vmovaps zmm4,zmm26
        vmovaps zmm17,zmm26

        vsubps  zmm2,zmm2,zmm22                 # vr = round(input / ln2)
        vsubps  zmm5,zmm5,zmm22
        vsubps  zmm18,zmm18,zmm22

        vfmadd231ps zmm0,zmm2,zmm24             # range reduce: vf -= (vr * ln2_high)
        vfmadd231ps zmm3,zmm5,zmm24
        vfmadd231ps zmm16,zmm18,zmm24

        vfmadd231ps zmm0,zmm2,zmm25             # range reduce: vf -= (vr * ln2_low)
        vfmadd231ps zmm3,zmm5,zmm25
        vfmadd231ps zmm16,zmm18,zmm25

        vfmadd213ps zmm1,zmm0,zmm27             # P0 * vf + P1
        vfmadd213ps zmm4,zmm3,zmm27
        vfmadd213ps zmm17,zmm16,zmm27
        vfmadd213ps zmm1,zmm0,zmm28             # * vf + P2
        vfmadd213ps zmm4,zmm3,zmm28
        vfmadd213ps zmm17,zmm16,zmm28
        vfmadd213ps zmm1,zmm0,zmm29             # * vf + P3
        vfmadd213ps zmm4,zmm3,zmm29
        vfmadd213ps zmm17,zmm16,zmm29
        vfmadd213ps zmm1,zmm0,zmm30             # * vf + P4
        vfmadd213ps zmm4,zmm3,zmm30
        vfmadd213ps zmm17,zmm16,zmm30
        vfmadd213ps zmm1,zmm0,zmm31             # * vf + P5
        vfmadd213ps zmm4,zmm3,zmm31
        vfmadd213ps zmm17,zmm16,zmm31
        vfmadd213ps zmm1,zmm0,zmm31             # * vf + P6
        vfmadd213ps zmm4,zmm3,zmm31
        vfmadd213ps zmm17,zmm16,zmm31

        vscalefps zmm1,zmm1,zmm2                # exp(ve) = exp(vf) * exp(vr)
        vscalefps zmm4,zmm4,zmm5
        vscalefps zmm17,zmm17,zmm18

        vaddps  zmm20,zmm20,zmm1                # accumulate exp() results to zmm20
        vaddps  zmm20,zmm20,zmm4
        vaddps  zmm20,zmm20,zmm17

        add     rcx,48*4                        # advance input by 48 elements

        // save to output
        vmovups ZMMWORD PTR [r8],zmm1
        vmovups ZMMWORD PTR [r8+64],zmm4
        vmovups ZMMWORD PTR [r8+128],zmm17

        add     r8,48*4                        # advance output by 48 elements

        sub     r9,48
        jae     .LSoftmaxStable.ComputeExpBy3x16

.LSoftmaxStable.ComputeExpByLT3x16:
        add     r9,48                          # correct for over-subtract above
        jz      .LSoftmaxStable.ReduceAccumulator
        mov     eax,-1
        kmovw   k1,eax                          # update mask to access all elements

.LSoftmaxStable.ComputeExpBy16Loop:
        cmp     r9,16
        jae     .LSoftmaxStable.ComputeExpBy16     # process 16
        lea     r10,C_UNDERSCORE(MlasOpmask16BitTableAvx512)[rip]
        kmovw   k1,WORD PTR [r10+r9*2]

.LSoftmaxStable.ComputeExpBy16:
        vaddps  zmm0{k1}{z},zmm19,ZMMWORD PTR [rcx]
                                                # bias by negative maximum value
        vmaxps  zmm0,zmm21,zmm0                 # clamp lower bound
        vmovaps zmm2,zmm23                      # 1/LN2
        vfmadd213ps zmm2,zmm0,zmm22             # (input / ln2) plus rounding bias
        vmovaps zmm1,zmm26                      # P0
        vsubps  zmm2,zmm2,zmm22                 # vr = round(input / ln2)
        vfmadd231ps zmm0,zmm2,zmm24             # range reduce: vf -= (vr * ln2_high)
        vfmadd231ps zmm0,zmm2,zmm25             # range reduce: vf -= (vr * ln2_low)
        vfmadd213ps zmm1,zmm0,zmm27             # P0 * vf + P1
        vfmadd213ps zmm1,zmm0,zmm28             # * vf + P2
        vfmadd213ps zmm1,zmm0,zmm29             # * vf + P3
        vfmadd213ps zmm1,zmm0,zmm30             # * vf + P4
        vfmadd213ps zmm1,zmm0,zmm31             # * vf + P5
        vfmadd213ps zmm1,zmm0,zmm31             # * vf + P6
        vscalefps zmm1,zmm1,zmm2                # exp(vf) * exp(vr)
        vaddps  zmm20{k1},zmm20,zmm1            # accumulate exp() results to zmm20

        add     rcx,16*4                        # advance input by 16 elements

        vmovups ZMMWORD PTR [r8]{k1},zmm1      # save result

        add     r8,16*4                        # advance output by 16 elements

        sub     r9,16
        ja      .LSoftmaxStable.ComputeExpBy16Loop

.LSoftmaxStable.ReduceAccumulator:
        vextractf64x4 ymm0,zmm20,1              # two 8 elements
        vaddps  zmm0,zmm0,zmm20                 # add to 8
        vhaddps ymm0,ymm0,ymm0                  # add two region
        vhaddps ymm0,ymm0,ymm0                  # add two region, all 4 elements with same sum
        vextractf128 xmm1,ymm0,1                # move 2nd region to xmm1
        vaddss  xmm0,xmm0,xmm1                  # add to get sum
        vrcp14ss xmm0,xmm0,xmm0                 # take reciprocal

/*=============== Loop3: normalize ==============*/

        # moving Input Output and N
        mov r8,rsi
        mov r9,rdx

        vbroadcastss ymm4,xmm0       # broadcast scale value
        cmp     r9,32
        jb      .LSoftmaxStable.ComputeSoftmaxBy8

.LSoftmaxStable.ComputeSoftmaxBy4x8:
        vmulps  ymm0,ymm4,YMMWORD PTR [r8]     # 1/sum(e^n)
        vmulps  ymm1,ymm4,YMMWORD PTR [r8+8*4]
        sub     r9,32                          # N-=32
        vmulps  ymm2,ymm4,YMMWORD PTR [r8+16*4]
        vmulps  ymm3,ymm4,YMMWORD PTR [r8+24*4]
        vmovups YMMWORD PTR [r8],ymm0          # save result
        vmovups YMMWORD PTR [r8+8*4],ymm1
        vmovups YMMWORD PTR [r8+16*4],ymm2
        vmovups YMMWORD PTR [r8+24*4],ymm3
        add     r8,32*4                        # advance output by 32 elements
        cmp     r9,32
        jae     .LSoftmaxStable.ComputeSoftmaxBy4x8

.LSoftmaxStable.ComputeSoftmaxBy8:
        cmp     r9,8
        jb      .LSoftmaxStable.ComputeSoftmaxByLT8
        vmulps  ymm0,ymm4,YMMWORD PTR [r8]     # 1/sum(e^n)
        sub     r9,8                           # N-=8
        vmovups YMMWORD PTR [r8],ymm0          # same result
        add     r8,8*4                         # advance output by 8 elements
        jmp     .LSoftmaxStable.ComputeSoftmaxBy8

.LSoftmaxStable.ComputeSoftmaxByLT8:
        test    r9,r9
        jz      .LSoftmaxStable.ExitKernel

.LSoftmaxStable.ComputeSoftmaxBy1:
        vmulss  xmm0,xmm4,DWORD PTR [r8]
        vmovss  DWORD PTR [r8],xmm0
        add     r8,4                           # advance output by 1 element
        dec     r9d                             # N-=1
        jnz     .LSoftmaxStable.ComputeSoftmaxBy1

.LSoftmaxStable.ExitKernel:
        vzeroupper
        ret
