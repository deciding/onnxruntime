/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SkipLayerNormAvx512F.s

Abstract:

    This module implements kernels for various transcendental functions.

    This implementation uses AVX512F instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

// constants

        .equ    SkipLayerNormCountN, 8
        .equ    SkipLayerNormSavedR12, -8
        .equ    SkipLayerNormSavedR13, -16

/*++

Routine Description:

    This routine implements a vectorized kernel for the exponential function.

Arguments:

    Input1 (rdi) - Supplies the input buffer, in this version, it needs to add bias

    Input2 (rsi) - Supplies the input buffer.

    Bias1 (rdx) - bias for input1

    Epsilon (xmm0) - Supplies float epsilon for variance

    gamma (rcx) - mul weights

    beta (r8) - add weights

    Output (r9) - Supplies the output buffer.

    N (mov to r11) - Supplies the number of elements to process.

Return Value:

    None.

--*/

        .globl  C_UNDERSCORE(MlasSkipLayerNormAvx512)
C_UNDERSCORE(MlasSkipLayerNormAvx512):

        mov     SkipLayerNormSavedR12[rsp],r12
        mov     SkipLayerNormSavedR13[rsp],r13

        vbroadcastss zmm11,xmm0                 # epsilon
        # TODO: ymm is enough for mean
        // vpbroadcastd zmm13,SkipLayerNormCountN[rsp] # broadcast countN
        // vcvtdq2ps    zmm13,zmm13                    # convert to float
        // vrcp28ps     zmm13,zmm13                    # calculate reciprocal
        mov          r11,SkipLayerNormCountN[rsp]
        movd         xmm13,r11
        cvtdq2ps     xmm13,xmm13
        vrcp14ss     xmm13,xmm13,xmm13                    # calculate reciprocal


// ================Stage 1: mean=================
        // r11 and r9 will be reused by different reduction loop
        // TODO: check whether need to recover r12
        // TODO: r11 conflict
        mov     r10,r11
        mov     r12,r9

        vpxord  zmm15,zmm15,zmm15                     # accum mean
        vpxord  zmm12,zmm12,zmm12                     # accum square mean

        sub     r10,64
        jb      .LLayerNorm.MeanReductionRemainingCount

.LLayerNorm.MeanReductionBy4x16Loop:
        vmovups  zmm0,ZMMWORD PTR [rdi]          # input1
        vmovups  zmm1,ZMMWORD PTR [rdi+64]
        vmovups  zmm2,ZMMWORD PTR [rdi+128]
        vmovups  zmm3,ZMMWORD PTR [rdi+192]

        vmovups  zmm4,ZMMWORD PTR [rsi]          # input2
        vmovups  zmm5,ZMMWORD PTR [rsi+64]
        vmovups  zmm6,ZMMWORD PTR [rsi+128]
        vmovups  zmm7,ZMMWORD PTR [rsi+192]

# TEST
        test     rdx,rdx
        jz       .LLayerNorm.AddInputs
        vaddps   zmm0,zmm0,ZMMWORD PTR [rdx]    # add bias
        vaddps   zmm1,zmm1,ZMMWORD PTR [rdx+64]
        vaddps   zmm2,zmm2,ZMMWORD PTR [rdx+128]
        vaddps   zmm3,zmm3,ZMMWORD PTR [rdx+192]
        add      rdx,64*4                        # advance bias by 64 elements

.LLayerNorm.AddInputs:
        vaddps   zmm0,zmm0,zmm4                # add two inputs
        vaddps   zmm1,zmm1,zmm5
        vaddps   zmm2,zmm2,zmm6
        vaddps   zmm3,zmm3,zmm7

        #vmovups   zmm28,ZMMWORD PTR [rdx]    # load bias
        #vmovups   zmm29,ZMMWORD PTR [rdx+64]
        #vmovups   zmm30,ZMMWORD PTR [rdx+128]
        #vmovups   zmm31,ZMMWORD PTR [rdx+192]

        #vaddps   zmm0,zmm0,zmm4                # add two inputs
        #vaddps   zmm1,zmm1,zmm5
        #vaddps   zmm2,zmm2,zmm6
        #vaddps   zmm3,zmm3,zmm7

        #vaddps   zmm0,zmm0,zmm28                # add bias
        #vaddps   zmm1,zmm1,zmm29
        #vaddps   zmm2,zmm2,zmm30
        #vaddps   zmm3,zmm3,zmm31
# -TEST

        vaddps   zmm15,zmm15,zmm0              # accumulate on zmm15
        vfmadd231ps   zmm12,zmm0,zmm0              # accumulate square on zmm12
        vaddps   zmm15,zmm15,zmm1
        vfmadd231ps   zmm12,zmm1,zmm1
        vaddps   zmm15,zmm15,zmm2
        vfmadd231ps   zmm12,zmm2,zmm2
        vaddps   zmm15,zmm15,zmm3
        vfmadd231ps   zmm12,zmm3,zmm3

        vmovups  ZMMWORD PTR [r12],zmm0          # save to output temporary, less data dependency
        vmovups  ZMMWORD PTR [r12+64],zmm1
        vmovups  ZMMWORD PTR [r12+128],zmm2
        vmovups  ZMMWORD PTR [r12+192],zmm3

        add     rdi,64*4                        # advance input1 by 64 elements
        add     rsi,64*4                        # advance input2 by 64 elements
        add     r12,64*4                         # advance output by 64 elements

        sub     r10,64                          # sub countN 64 elements
        jae     .LLayerNorm.MeanReductionBy4x16Loop

.LLayerNorm.MeanReductionRemainingCount:
        add     r10,64                          # correct for over-subtract above
        jz      .LLayerNorm.MeanReduceAccumulator
        mov     eax,-1
        kmovw   k1,eax                          # update mask to access all elements

.LLayerNorm.MeanReductionBy16Loop:
        cmp     r10,16
        jae     .LLayerNorm.MeanReductionBy16
        lea     r13,C_UNDERSCORE(MlasOpmask16BitTableAvx512)[rip] # can move this to init, but this is hardly used
        kmovw   k1,WORD PTR [r13+r10*2]

.LLayerNorm.MeanReductionBy16:
        vmovups  zmm0{k1}{z},ZMMWORD PTR [rdi]          # input1
        vmovups  zmm4{k1}{z},ZMMWORD PTR [rsi]          # input2

        test     rdx,rdx
        jz       .LLayerNorm.AddInputsBy16
        vaddps   zmm0{k1}{z},zmm0,ZMMWORD PTR [rdx]    # add bias
        add      rdx,16*4                        # advance bias by 16 elements
.LLayerNorm.AddInputsBy16:
        vaddps   zmm0,zmm0,zmm4                # add two inputs
        vaddps   zmm15,zmm15,zmm0              # accumulate on zmm15
        vfmadd231ps   zmm12,zmm0,zmm0              # accumulate square on zmm12
        vmovups  ZMMWORD PTR [r12]{k1},zmm0          # save to output temporary, less data dependency

        add     rdi,16*4                        # advance input1 by 16 elements
        add     rsi,16*4                        # advance input2 by 16 elements
        add     r12,16*4                         # advance output by 16 elements

        sub     r10,16                          # sub countN 16 elements
        ja      .LLayerNorm.MeanReductionBy16Loop # not jae because go to accumulator if equal to 0

.LLayerNorm.MeanReduceAccumulator:
        vextractf64x4 ymm14,zmm15,1     # extract 8 elements from upper part
        vaddps  zmm15,zmm15,zmm14	# here only the lower 8 elements are useful
        vhaddps ymm15,ymm15,ymm15	# accum half of each of the two region
        vhaddps ymm15,ymm15,ymm15	# accum all 4 elements of the two regions

        vextractf64x4 ymm14,zmm12,1     # extract 8 elements from upper part
        vaddps  zmm12,zmm12,zmm14	# here only the lower 8 elements are useful
        vhaddps ymm12,ymm12,ymm12	# accum half of each of the two region
        vhaddps ymm12,ymm12,ymm12	# accum all 4 elements of the two regions

        vextractf128 xmm14,ymm15,1      # mov second region to xmm14
        vaddss  xmm15,xmm15,xmm14       # add the two elements from the two region
        vmulss  xmm15,xmm15,xmm13       # sum to mean
        vbroadcastss zmm15,xmm15

        vextractf128 xmm14,ymm12,1      # mov second region to xmm14
        vaddss  xmm12,xmm12,xmm14       # add the two elements from the two region
        vfmsub231ss  xmm11,xmm15,xmm15  # mean**2 - epsilon
        vfmsub213ss  xmm12,xmm13,xmm11  # sumvar/countN - mean**2 + epsilon
        vsqrtss      xmm12,xmm12,xmm12  # sqrt
        vbroadcastss zmm12,xmm12


// ================Stage 2: affine=================

        // recover countN and output
        mov     r10,r11
        mov     r12,r9

        sub     r10,64
        jb      .LLayerNorm.AffineRemainingCount

.LLayerNorm.AffineBy4x16Loop:
        vmovups  zmm0,ZMMWORD PTR [r12]          # temp output
        vmovups  zmm1,ZMMWORD PTR [r12+64]
        vmovups  zmm2,ZMMWORD PTR [r12+128]
        vmovups  zmm3,ZMMWORD PTR [r12+192]

        vsubps   zmm0,zmm0,zmm15                # sub mean
        vsubps   zmm1,zmm1,zmm15
        vsubps   zmm2,zmm2,zmm15
        vsubps   zmm3,zmm3,zmm15

        vmovups  zmm4,ZMMWORD PTR [rcx]          # gamma
        vmovups  zmm5,ZMMWORD PTR [rcx+64]
        vmovups  zmm6,ZMMWORD PTR [rcx+128]
        vmovups  zmm7,ZMMWORD PTR [rcx+192]

        vdivps   zmm0,zmm0,zmm12                # div stdev
        vdivps   zmm1,zmm1,zmm12
        vdivps   zmm2,zmm2,zmm12
        vdivps   zmm3,zmm3,zmm12

        vfmadd213ps   zmm0,zmm4,ZMMWORD PTR [r8]                # *gamma+beta
        vfmadd213ps   zmm1,zmm5,ZMMWORD PTR [r8+64]
        vfmadd213ps   zmm2,zmm6,ZMMWORD PTR [r8+128]
        vfmadd213ps   zmm3,zmm7,ZMMWORD PTR [r8+192]

        vmovups  ZMMWORD PTR [r12],zmm0          # save to output temporary
        vmovups  ZMMWORD PTR [r12+64],zmm1
        vmovups  ZMMWORD PTR [r12+128],zmm2
        vmovups  ZMMWORD PTR [r12+192],zmm3

        add     rcx,64*4                         # advance gamma by 64 elements
        add     r8,64*4                          # advance beta by 64 elements
        add     r12,64*4                         # advance output by 64 elements

        sub     r10,64                          # sub countN 64 elements
        jae     .LLayerNorm.AffineBy4x16Loop

.LLayerNorm.AffineRemainingCount:
        add     r10,64                          # correct for over-subtract above
        jz      .LLayerNorm.ExitKernel
        mov     eax,-1
        kmovw   k1,eax                          # update mask to access all elements

.LLayerNorm.AffineBy16Loop:
        cmp     r10,16
        jae     .LLayerNorm.AffineBy16
        lea     r13,C_UNDERSCORE(MlasOpmask16BitTableAvx512)[rip] # can move this to init, but this is hardly used
        kmovw   k1,WORD PTR [r13+r10*2]

.LLayerNorm.AffineBy16:
        vmovups  zmm0{k1}{z},ZMMWORD PTR [r12]          # temp output
        vsubps   zmm0,zmm0,zmm15                # sub mean
        vmovups  zmm4{k1}{z},ZMMWORD PTR [rcx]          # gamma
        vdivps   zmm0,zmm0,zmm12                # div stdev
        vfmadd213ps   zmm0{k1}{z},zmm4,ZMMWORD PTR [r8]                # *gamma+beta
        vmovups  ZMMWORD PTR [r12]{k1},zmm0          # save to output temporary

        add     rcx,16*4                         # advance gamma by 16 elements
        add     r8,16*4                          # advance beta by 16 elements
        add     r12,16*4                         # advance output by 16 elements

        sub     r10,16                          # sub countN 16 elements
        ja      .LLayerNorm.AffineBy16Loop      # not jae because go to accumulator if equal to 0

.LLayerNorm.ExitKernel:

        mov     r12,SkipLayerNormSavedR12[rsp]
        mov     r13,SkipLayerNormSavedR13[rsp]
        vzeroupper
        ret

        .end
