/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    BiasGeluAvx512.s

Abstract:

    This module implements a kernel for computing the error function for a
    buffer of elements.

    This implementation uses AVX fused multiply/add instructions.

    This is taken from njuffa's implementation https://stackoverflow.com/a/35148199/8470034
    The onnxruntime author is tracysh

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

//
// Structure layout for the erf constants block.
//

        .equ    ErfUpperAbsRange, 0
        .equ    ErfSplitBoundary, 4
        .equ    ErfSMALL_P0, 8
        .equ    ErfSMALL_P1, 12
        .equ    ErfSMALL_P2, 16
        .equ    ErfSMALL_P3, 20
        .equ    ErfSMALL_P4, 24
        .equ    ErfSMALL_P5_Minus_One, 28
        .equ    ErfReserve0, 32
        .equ    ErfBIG_P0, 36
        .equ    ErfBIG_P1, 40
        .equ    ErfBIG_P2, 44
        .equ    ErfBIG_P3, 48
        .equ    ErfBIG_P4, 52
        .equ    ErfBIG_P5, 56
        .equ    ErfBIG_P6_Minus_One, 60
        .equ    ErfNegZero, 64
        .equ    ErfOne, 68

        .equ    ExpConstOffset, 72
        .equ    Exp_UpperRange, 0 + ExpConstOffset
        .equ    Exp_LowerRange, 4 + ExpConstOffset
        .equ    Exp_Log2Reciprocal, 8 + ExpConstOffset
        .equ    Exp_log2_hi, 12 + ExpConstOffset
        .equ    Exp_log2_lo, 16 + ExpConstOffset
        .equ    Exp_P0, 20 + ExpConstOffset
        .equ    Exp_P1, 24 + ExpConstOffset
        .equ    Exp_P2, 28 + ExpConstOffset
        .equ    Exp_P3, 32 + ExpConstOffset
        .equ    Exp_P4, 36 + ExpConstOffset
        .equ    Exp_P5, 40 + ExpConstOffset
        .equ    Exp_P6, 44 + ExpConstOffset
        .equ    Exp_C, 48 + ExpConstOffset
        .equ    Exp_X7F, 52 + ExpConstOffset

// Misc constant
        .equ    Misc_SQRT1_2, 0
        .equ    Misc_HALF, 4

//
// Stack frame layout for the erf kernel.
//
// CHANGE ZMM, TODO: change name
        .equ    BiasGeluBuffer0, 0
        .equ    BiasGeluBuffer1, 256
        .equ    BiasGelu_CountN, 512
        .equ    BiasGelu_ReturnAddress, 512+8

/*++

MLAS_INTERNAL_DATA const struct {
    float ErfUpperAbsRange;              3.925f,                              
    float ErfSplitBoundary;              0.921875f,                           
    float ErfSMALL_P0;                   -5.99104969e-4f,                
    float ErfSMALL_P1;                   4.99339588e-3f,                 
    float ErfSMALL_P2;                   -2.67667342e-2f,                
    float ErfSMALL_P3;                   1.12818025e-1f,                 
    float ErfSMALL_P4;                   -3.76124859e-1f,                
    float ErfSMALL_P5_Minus_One;         1.28379151e-1f,                           
    float ErfReserved0;                  0.0f,                            
    float ErfBIG_P0;                     1.72948930e-5f,               
    float ErfBIG_P1;                     -3.83208680e-4f,              
    float ErfBIG_P2;                     3.88393435e-3f,               
    float ErfBIG_P3;                     -2.42545605e-2f,              
    float ErfBIG_P4;                     1.06777847e-1f,               
    float ErfBIG_P5;                     6.34846687e-1f,               
    float ErfBIG_P6_Minus_One;           1.28717512e-1f,                         
    float ErfNegZero;                    -0.0f,                         
    float ErfOne;                        1.0f,                      
                                                                 
    float Exp_UpperRange;                88.3762626647950f,                               
    float Exp_LowerRange;                -88.3762626647949f,                                      
    float Exp_Log2Reciprocal;            1.44269504088896341f,       1/LN2
    float Exp_log2_hi;                   -6.93145752e-1f,                
    float Exp_log2_lo;                   -1.42860677e-6f,                
    float Exp_P0;                        1.38319808e-3f,            
    float Exp_P1;                        8.37550033e-3f,            
    float Exp_P2;                        4.16689515e-2f,            
    float Exp_P3;                        1.66664466e-1f,            
    float Exp_P4;                        4.99999851e-1f,            
    float Exp_P5;                        1.00000000e+0f,            
    float Exp_P6;                        1.00000000e+0f,            
    float Exp_C;                         1.25829120e+7f,           
    int32_t Exp_X7F;                     127,                                                      
}                            
                            
Routine Description:                            
                            
    This routine implements a vectorized kernel for the error function.                            
                            
Arguments:

    Input (rdi) - Supplies the input buffer.

    Output (rsi) - Supplies the output buffer.

    N (rdx)  - Supplies the number of elements to process.

    Bias (rcx) - Supplies the bias buffer.

Return Value:

    None.

--*/

        .globl  C_UNDERSCORE(MlasBiasGeluAvx512)
C_UNDERSCORE(MlasBiasGeluAvx512):
        sub     rsp,BiasGelu_ReturnAddress # get stack space for 2 buffers
        lea     rax,C_UNDERSCORE(MlasErfConstants)[rip] # rax now becomes the pointer to const
        lea     r11,C_UNDERSCORE(MlasMiscConstants)[rip] # r11 now becomes the pointer to  misc const

        sub     rdx,16*4 # 4x8 loop minus on N, 4 reg each with 16 elements
        jb      .LBiasGeluProcessRemainingCount

.LComputeBGBias4x16Loop:
        vbroadcastss zmm15,ErfNegZero[rax]      # broad cast -0.0 to zmm15 8 vec, 100000...
        vbroadcastss zmm27,Misc_SQRT1_2[r11]      # broad cast 0.707... to zmm27 16 vec
        vbroadcastss zmm26,Misc_HALF[r11]      # broad cast 0.5 to zmm26 16 vec
        //vmovups zmm0,ZMMWORD PTR [rdi]          # original input vx0
        //vmovups zmm1,ZMMWORD PTR [rdi+64]       # original input vx1
        //vmovups zmm2,ZMMWORD PTR [rdi+128]       # original input vx2
        //vmovups zmm3,ZMMWORD PTR [rdi+192]       # original input vx3
        vmovups zmm16,ZMMWORD PTR [rdi]          # original input vx0
        vmovups zmm17,ZMMWORD PTR [rdi+64]       # original input vx1
        vmovups zmm18,ZMMWORD PTR [rdi+128]       # original input vx2
        vmovups zmm19,ZMMWORD PTR [rdi+192]       # original input vx3

        test   rcx,rcx
        jz     .LComputeSqrtBG
        //vaddps zmm16,zmm0,ZMMWORD PTR [rcx]          # add bias
        //vaddps zmm17,zmm1,ZMMWORD PTR [rcx+64]
        //vaddps zmm18,zmm2,ZMMWORD PTR [rcx+128]
        //vaddps zmm19,zmm3,ZMMWORD PTR [rcx+192]
        vaddps zmm16,zmm16,ZMMWORD PTR [rcx]          # add bias
        vaddps zmm17,zmm17,ZMMWORD PTR [rcx+64]
        vaddps zmm18,zmm18,ZMMWORD PTR [rcx+128]
        vaddps zmm19,zmm19,ZMMWORD PTR [rcx+192]
        add    rcx,64*4

.LComputeSqrtBG:
        vmulps zmm0,zmm16,zmm27          # * sqrt(1/2)
        vmulps zmm1,zmm17,zmm27
        vmulps zmm2,zmm18,zmm27
        vmulps zmm3,zmm19,zmm27

        vmulps zmm20,zmm16,zmm26          # * 0.5 for residual
        vmulps zmm21,zmm17,zmm26
        vmulps zmm22,zmm18,zmm26
        vmulps zmm23,zmm19,zmm26

.LComputeBGErf4x16Loop:

        vandps  zmm4,zmm0,zmm15                 # vsign0
        vandps  zmm5,zmm1,zmm15                 # vsign1
        vandps  zmm6,zmm2,zmm15                 # vsign2
        vandps  zmm7,zmm3,zmm15                 # vsign3
        vandnps zmm0,zmm15,zmm0                 # abs(vx0)  va0, 0111111...
        vandnps zmm1,zmm15,zmm1                 # abs(vx1)  va1
        vandnps zmm2,zmm15,zmm2                 # abs(vx2)  va2
        vandnps zmm3,zmm15,zmm3                 # abs(vx3)  va3

        vbroadcastss zmm14,ErfUpperAbsRange[rax] # 3.925f, ERF floating point upper limit
        vmovups ZMMWORD PTR BiasGeluBuffer0[rsp],zmm4        # save sign to buffer0
        vmovups ZMMWORD PTR BiasGeluBuffer0[rsp+64],zmm5
        vmovups ZMMWORD PTR BiasGeluBuffer0[rsp+128],zmm6
        vmovups ZMMWORD PTR BiasGeluBuffer0[rsp+192],zmm7

        vbroadcastss zmm8,ErfSMALL_P0[rax]      # -5.99104969e-4f, prefetch for lower data dependency
        vminps  zmm0,zmm0,zmm14                 # force abs value in range
        vminps  zmm1,zmm1,zmm14
        vminps  zmm2,zmm2,zmm14
        vminps  zmm3,zmm3,zmm14
        vmovaps zmm9,zmm8       # zmm 8 9 10 11 init to P0
        vmovaps zmm10,zmm8
        vmovaps zmm11,zmm8

        vbroadcastss zmm15,ErfSMALL_P1[rax]     # 4.99339588e-3f
        vmulps  zmm4,zmm0,zmm0                  # vs0 (square), zmm 0 1 2 3 assign to t, zmm 4 5 6 7 assign to s
        vmulps  zmm5,zmm1,zmm1                  # vs1
        vmulps  zmm6,zmm2,zmm2                  # vs2
        vmulps  zmm7,zmm3,zmm3                  # vs3

        vbroadcastss zmm14,ErfSMALL_P2[rax]     # -2.67667342e-2f
        vfmadd213ps zmm8,zmm4,zmm15             # P0 * s + P1
        vfmadd213ps zmm9,zmm5,zmm15
        vfmadd213ps zmm10,zmm6,zmm15
        vfmadd213ps zmm11,zmm7,zmm15

        vbroadcastss zmm13,ErfSMALL_P3[rax]     # 1.12818025e-1f
        vfmadd213ps zmm8,zmm4,zmm14             # * s + P2
        vfmadd213ps zmm9,zmm5,zmm14
        vfmadd213ps zmm10,zmm6,zmm14
        vfmadd213ps zmm11,zmm7,zmm14

        vbroadcastss zmm15,ErfSMALL_P4[rax]     # -3.76124859e-1f
        vfmadd213ps zmm8,zmm4,zmm13             # * s + P3
        vfmadd213ps zmm9,zmm5,zmm13
        vfmadd213ps zmm10,zmm6,zmm13
        vfmadd213ps zmm11,zmm7,zmm13

        vbroadcastss zmm14,ErfSMALL_P5_Minus_One[rax] # 1.28379151e-1f
        vfmadd213ps zmm8,zmm4,zmm15             # * s + P4
        vfmadd213ps zmm9,zmm5,zmm15
        vfmadd213ps zmm10,zmm6,zmm15
        vfmadd213ps zmm11,zmm7,zmm15

        vfmadd213ps zmm8,zmm4,zmm14             # * s + P5 - 1
        vfmadd213ps zmm9,zmm5,zmm14
        vfmadd213ps zmm10,zmm6,zmm14
        vfmadd213ps zmm11,zmm7,zmm14

        vbroadcastss zmm12,ErfSplitBoundary[rax]        # 0.921875f
        vfmadd213ps zmm8,zmm0,zmm0              # * t + t
        vfmadd213ps zmm9,zmm1,zmm1
        vfmadd213ps zmm10,zmm2,zmm2
        vfmadd213ps zmm11,zmm3,zmm3

        # old school
        #vcmpgtps zmm4,zmm0,zmm12                # vmask0, greater than boundary 1111... else 00000...
        #vcmpgtps zmm5,zmm1,zmm12                # vmask1
        #vcmpgtps zmm6,zmm2,zmm12                # vmask2
        #vcmpgtps zmm7,zmm3,zmm12                # vmask3

        #vandnps zmm8,zmm4,zmm8                  # only keep smaller than boundary, otherwise 0
        #vandnps zmm9,zmm5,zmm9
        #vandnps zmm10,zmm6,zmm10
        #vandnps zmm11,zmm7,zmm11

        # avx512
        vcmpleps k1,zmm0,zmm12                # vmask0, greater than boundary 1 else 0
        vcmpleps k2,zmm1,zmm12                # vmask1
        vcmpleps k3,zmm2,zmm12                # vmask2
        vcmpleps k4,zmm3,zmm12                # vmask3

        vbroadcastss zmm15,ErfBIG_P1[rax]       # -3.83208680e-4f
        vmovups ZMMWORD PTR BiasGeluBuffer1[rsp],zmm8        # keep the smaller number result in buffer1
        vmovups ZMMWORD PTR BiasGeluBuffer1[rsp+64],zmm9
        vmovups ZMMWORD PTR BiasGeluBuffer1[rsp+128],zmm10
        vmovups ZMMWORD PTR BiasGeluBuffer1[rsp+192],zmm11

.BiggerNumbers512:
        vbroadcastss zmm8,ErfBIG_P0[rax]        # 1.72948930e-5f
        # leave this to merge phase
        #vandps  zmm0,zmm4,zmm0                  # if final result is smaller, mask them out
        #vandps  zmm1,zmm5,zmm1
        #vandps  zmm2,zmm6,zmm2
        #vandps  zmm3,zmm7,zmm3
        vmovaps zmm9,zmm8                       # zmm 0 1 2 3 assign to t, zmm 8 9 10 11 assign to P0
        vmovaps zmm10,zmm8
        vmovaps zmm11,zmm8

        vbroadcastss zmm14,ErfBIG_P2[rax]       # 3.88393435e-3f
        vfmadd213ps zmm8,zmm0,zmm15             # P0 * t + P1
        vfmadd213ps zmm9,zmm1,zmm15
        vfmadd213ps zmm10,zmm2,zmm15
        vfmadd213ps zmm11,zmm3,zmm15

        vbroadcastss zmm13,ErfBIG_P3[rax]       # -2.42545605e-2f
        vfmadd213ps zmm8,zmm0,zmm14             # * t + P2
        vfmadd213ps zmm9,zmm1,zmm14
        vfmadd213ps zmm10,zmm2,zmm14
        vfmadd213ps zmm11,zmm3,zmm14

        vbroadcastss zmm15,ErfBIG_P4[rax]       # 1.06777847e-1f
        vfmadd213ps zmm8,zmm0,zmm13             # * t + P3
        vfmadd213ps zmm9,zmm1,zmm13
        vfmadd213ps zmm10,zmm2,zmm13
        vfmadd213ps zmm11,zmm3,zmm13

        vbroadcastss zmm14,ErfBIG_P5[rax]       # 6.34846687e-1f
        vfmadd213ps zmm8,zmm0,zmm15             # * t + P4
        vfmadd213ps zmm9,zmm1,zmm15
        vfmadd213ps zmm10,zmm2,zmm15
        vfmadd213ps zmm11,zmm3,zmm15

        vbroadcastss zmm13,ErfBIG_P6_Minus_One[rax]     # 1.28717512e-1f
        vfmadd213ps zmm8,zmm0,zmm14             # * t + P5
        vfmadd213ps zmm9,zmm1,zmm14
        vfmadd213ps zmm10,zmm2,zmm14
        vfmadd213ps zmm11,zmm3,zmm14

        vbroadcastss zmm15,ErfNegZero[rax]
        vfmadd213ps zmm8,zmm0,zmm13             # * t + P6 - 1
        vfmadd213ps zmm9,zmm1,zmm13
        vfmadd213ps zmm10,zmm2,zmm13
        vfmadd213ps zmm11,zmm3,zmm13

        vbroadcastss zmm14,Exp_LowerRange[rax]  # -88.3762626647949f
        vfmadd213ps zmm8,zmm0,zmm0              # * t + t
        vfmadd213ps zmm9,zmm1,zmm1
        vfmadd213ps zmm10,zmm2,zmm2
        vfmadd213ps zmm11,zmm3,zmm3

        vbroadcastss zmm4,Exp_Log2Reciprocal[rax]       # 1/LN2
        vxorps  zmm8,zmm8,zmm15         # take negative
        vxorps  zmm9,zmm9,zmm15
        vxorps  zmm10,zmm10,zmm15
        vxorps  zmm11,zmm11,zmm15

        # START expf(zmm8 -- zmm11)
        vbroadcastss zmm13,Exp_C[rax]           # 1.25829120e+7f
        vmovaps zmm5,zmm4               # assign to 1/LN2
        vmovaps zmm6,zmm4
        vmovaps zmm7,zmm4

        # clip exp lower,  zmm 8 9 10 11 renamed as ve
        vmaxps  zmm8,zmm8,zmm14
        vmaxps  zmm9,zmm9,zmm14
        vmaxps  zmm10,zmm10,zmm14
        vmaxps  zmm11,zmm11,zmm14

        vbroadcastss zmm0,Exp_log2_hi[rax]      # -6.93145752e-1f
        vfmadd213ps zmm4,zmm8,zmm13             # 1/LN2 * ve + 0x4b4000...
        vfmadd213ps zmm5,zmm9,zmm13
        vfmadd213ps zmm6,zmm10,zmm13
        vfmadd213ps zmm7,zmm11,zmm13

        vbroadcastss zmm15,Exp_log2_lo[rax]     # -1.42860677e-6f
        vmovaps zmm1,zmm0                       # assign to -C1
        vmovaps zmm2,zmm0
        vmovaps zmm3,zmm0

        vsubps  zmm4,zmm4,zmm13                 # -0x4b4000..., zmm 4 5 6 7 assign to vr = round(), integer part
        vsubps  zmm5,zmm5,zmm13
        vsubps  zmm6,zmm6,zmm13
        vsubps  zmm7,zmm7,zmm13

        # zmm 0 1 2 3: vf, 4 5 6 7: vr, 8 9 10 11: ve
        vfmadd213ps zmm0,zmm4,zmm8              # vf = ve - C1 * vr
        vfmadd213ps zmm1,zmm5,zmm9
        vfmadd213ps zmm2,zmm6,zmm10
        vfmadd213ps zmm3,zmm7,zmm11

        vbroadcastss zmm8,Exp_P0[rax]           # 1.38319808e-3f
        vfmadd231ps zmm0,zmm4,zmm15             # vf -= C2 * vr
        vfmadd231ps zmm1,zmm5,zmm15
        vfmadd231ps zmm2,zmm6,zmm15
        vfmadd231ps zmm3,zmm7,zmm15
        vmovaps zmm9,zmm8
        vmovaps zmm10,zmm8
        vmovaps zmm11,zmm8

        vbroadcastss zmm14,Exp_P1[rax]
        vbroadcastss zmm13,Exp_P2[rax]
        vfmadd213ps zmm8,zmm0,zmm14             # P0 * vf + P1
        vfmadd213ps zmm9,zmm1,zmm14
        vfmadd213ps zmm10,zmm2,zmm14
        vfmadd213ps zmm11,zmm3,zmm14

        vbroadcastss zmm12,Exp_P3[rax]
        vfmadd213ps zmm8,zmm0,zmm13             # * vf + P2
        vfmadd213ps zmm9,zmm1,zmm13
        vfmadd213ps zmm10,zmm2,zmm13
        vfmadd213ps zmm11,zmm3,zmm13

        vbroadcastss zmm15,Exp_P4[rax]
        vfmadd213ps zmm8,zmm0,zmm12             # * vf + P3
        vfmadd213ps zmm9,zmm1,zmm12
        vfmadd213ps zmm10,zmm2,zmm12
        vfmadd213ps zmm11,zmm3,zmm12

        vbroadcastss zmm14,Exp_P5[rax]
        vfmadd213ps zmm8,zmm0,zmm15             # * vf + P4
        vfmadd213ps zmm9,zmm1,zmm15
        vfmadd213ps zmm10,zmm2,zmm15
        vfmadd213ps zmm11,zmm3,zmm15

        vbroadcastss zmm13,Exp_P6[rax]
        vfmadd213ps zmm8,zmm0,zmm14             # * vf + P5
        vfmadd213ps zmm9,zmm1,zmm14
        vfmadd213ps zmm10,zmm2,zmm14
        vfmadd213ps zmm11,zmm3,zmm14

        vbroadcastss zmm12,Exp_X7F[rax]         # 0x7f 127
        vfmadd213ps zmm8,zmm0,zmm13             # * vf + P6
        vfmadd213ps zmm9,zmm1,zmm13
        vfmadd213ps zmm10,zmm2,zmm13
        vfmadd213ps zmm11,zmm3,zmm13

        vcvttps2dq zmm4,zmm4                    # convert vr to int
        vcvttps2dq zmm5,zmm5
        vcvttps2dq zmm6,zmm6
        vcvttps2dq zmm7,zmm7


        vbroadcastss zmm15,ErfOne[rax]
        vpaddd  zmm4,zmm4,zmm12                 # vr +127 to become the exponent in FP32
        vpaddd  zmm5,zmm5,zmm12
        vpaddd  zmm6,zmm6,zmm12
        vpaddd  zmm7,zmm7,zmm12

        vpslld  zmm4,zmm4,23                    # shift left 23, move vr to the correct bit, this is the FP32 representation of the int part
        vpslld  zmm5,zmm5,23
        vpslld  zmm6,zmm6,23
        vpslld  zmm7,zmm7,23

        vmulps  zmm8,zmm8,zmm4                  # exp(vr) * exp(vf)
        vmulps  zmm9,zmm9,zmm5
        vmulps  zmm10,zmm10,zmm6
        vmulps  zmm11,zmm11,zmm7

        vsubps  zmm8,zmm15,zmm8                 # 1 - exp(ve)
        vsubps  zmm9,zmm15,zmm9
        vsubps  zmm10,zmm15,zmm10
        vsubps  zmm11,zmm15,zmm11

        # merge small numbers' result
        #vorps   zmm8,zmm8,ZMMWORD PTR ErfBuffer1_512[rsp]
        #vorps   zmm9,zmm9,ZMMWORD PTR ErfBuffer1_512[rsp+64]
        #vorps   zmm10,zmm10,ZMMWORD PTR ErfBuffer1_512[rsp+128]
        #vorps   zmm11,zmm11,ZMMWORD PTR ErfBuffer1_512[rsp+192]
        vmovups zmm8{k1}, ZMMWORD PTR BiasGeluBuffer1[rsp]
        vmovups zmm9{k2}, ZMMWORD PTR BiasGeluBuffer1[rsp+64]
        vmovups zmm10{k3}, ZMMWORD PTR BiasGeluBuffer1[rsp+128]
        vmovups zmm11{k4}, ZMMWORD PTR BiasGeluBuffer1[rsp+192]

        # copy sign
        vorps   zmm0,zmm8,ZMMWORD PTR BiasGeluBuffer0[rsp]
        vorps   zmm1,zmm9,ZMMWORD PTR BiasGeluBuffer0[rsp+64]
        vorps   zmm2,zmm10,ZMMWORD PTR BiasGeluBuffer0[rsp+128]
        vorps   zmm3,zmm11,ZMMWORD PTR BiasGeluBuffer0[rsp+192]

.LComputeResidual4x16Loop:

        vfmadd213ps zmm0,zmm20,zmm20 # * temp + temp
        vfmadd213ps zmm1,zmm21,zmm21
        vfmadd213ps zmm2,zmm22,zmm22
        vfmadd213ps zmm3,zmm23,zmm23

        # save the results
        vmovups ZMMWORD PTR [rsi],zmm0
        vmovups ZMMWORD PTR [rsi+64],zmm1
        vmovups ZMMWORD PTR [rsi+128],zmm2
        vmovups ZMMWORD PTR [rsi+192],zmm3

        add     rdi,64*4                        # advance by 4*16 elements
        add     rsi,64*4
        sub     rdx,64
        jae     .LComputeBGBias4x16Loop

.LBiasGeluProcessRemainingCount:
        add     rdx,64                          # correct for over-subtract above
        jz      .LBiasGeluExit                # jump directly to result if no N left

.LComputeBGBias1x16Loop:
        vbroadcastss zmm15,ErfNegZero[rax]
        vbroadcastss zmm27,Misc_SQRT1_2[r11]      # broad cast 0.707... to zmm27 16 vec
        vbroadcastss zmm26,Misc_HALF[r11]      # broad cast 0.5 to zmm26 16 vec

        # AVX512 version of masked load
        lea     r10,C_UNDERSCORE(MlasOpmask16BitTableAvx512)[rip]
        kmovw   k5,WORD PTR [r10+rdx*2] # rdx * 2byte
        //vmovups zmm0{k5}{z},ZMMWORD PTR [rdi]          # original input vx0
        vmovups zmm16{k5}{z},ZMMWORD PTR [rdi]          # original input vx0

        test   rcx,rcx
        jz     .LComputeSqrtBG1x16
        //vaddps zmm16{k5}{z},zmm0,ZMMWORD PTR [rcx]          # add bias
        vaddps zmm16{k5}{z},zmm16,ZMMWORD PTR [rcx]          # add bias
        add     rcx,16*4

.LComputeSqrtBG1x16:
        vmulps zmm0,zmm16,zmm27          # * sqrt(1/2)

        vmulps zmm20,zmm16,zmm26          # * 0.5 for residual

.LComputeBGErf1x16Loop:

        vandps  zmm4,zmm0,zmm15                 # vsign0
        vandnps zmm0,zmm15,zmm0                 # abs(vx0)  va0

        vbroadcastss zmm14,ErfUpperAbsRange[rax]
        vmovups ZMMWORD PTR BiasGeluBuffer0[rsp],zmm4        # save sign to buffer0

        vbroadcastss zmm8,ErfSMALL_P0[rax]
        vminps  zmm0,zmm0,zmm14                 # force abs value in range, t

        vbroadcastss zmm15,ErfSMALL_P1[rax]
        vmulps  zmm4,zmm0,zmm0                  # vs0 (square), s

        vbroadcastss zmm14,ErfSMALL_P2[rax]
        vfmadd213ps zmm8,zmm4,zmm15             # P0 * s + P1

        vbroadcastss zmm13,ErfSMALL_P3[rax]
        vfmadd213ps zmm8,zmm4,zmm14             # * s + P2

        vbroadcastss zmm15,ErfSMALL_P4[rax]
        vfmadd213ps zmm8,zmm4,zmm13             # * s + P3

        vbroadcastss zmm14,ErfSMALL_P5_Minus_One[rax]
        vfmadd213ps zmm8,zmm4,zmm15             # * s + P4

        vfmadd213ps zmm8,zmm4,zmm14             # * s + P5 - 1

        vbroadcastss zmm12,ErfSplitBoundary[rax]
        vfmadd213ps zmm8,zmm0,zmm0              # * t + t

        vcmpleps k1,zmm0,zmm12                # vmask0

        #vandnps zmm8,zmm4,zmm8                  # only reserve smaller number

        vmovups ZMMWORD PTR BiasGeluBuffer1[rsp],zmm8 # save smaller number to buffer 1

.BiggerNumbersRemaining512:
        vbroadcastss zmm15,ErfBIG_P1[rax]
        vbroadcastss zmm8,ErfBIG_P0[rax]
        #vandps  zmm0,zmm4,zmm0                  # mask the bigger number t

        vbroadcastss zmm14,ErfBIG_P2[rax]
        vfmadd213ps zmm8,zmm0,zmm15             # P0 * t + P1

        vbroadcastss zmm13,ErfBIG_P3[rax]
        vfmadd213ps zmm8,zmm0,zmm14             # * t + P2

        vbroadcastss zmm15,ErfBIG_P4[rax]
        vfmadd213ps zmm8,zmm0,zmm13             # * t + P3

        vbroadcastss zmm14,ErfBIG_P5[rax]
        vfmadd213ps zmm8,zmm0,zmm15             # * t + P4

        vbroadcastss zmm13,ErfBIG_P6_Minus_One[rax]
        vfmadd213ps zmm8,zmm0,zmm14             # * t + P5

        vbroadcastss zmm15,ErfNegZero[rax]
        vfmadd213ps zmm8,zmm0,zmm13             # * t + P6 - 1

        vbroadcastss zmm14,Exp_LowerRange[rax]
        vfmadd213ps zmm8,zmm0,zmm0              # * t + t

        vbroadcastss zmm4,Exp_Log2Reciprocal[rax]
        vxorps  zmm8,zmm8,zmm15                 # take negation

        # expf(zmm8 -- zmm11)
        vbroadcastss zmm13,Exp_C[rax]

        vmaxps  zmm8,zmm8,zmm14                 # clip, zmm8 known as ve

        vbroadcastss zmm0,Exp_log2_hi[rax]
        vfmadd213ps zmm4,zmm8,zmm13             # 1/LN2 * ve + 0x4b400...

        vbroadcastss zmm15,Exp_log2_lo[rax]

        vsubps  zmm4,zmm4,zmm13                 # -0x4b400..., zmm4 to vr = round()

        vfmadd213ps zmm0,zmm4,zmm8              # vf = ve - C1 * vr 

        vbroadcastss zmm8,Exp_P0[rax]

        vfmadd231ps zmm0,zmm4,zmm15             # vf -= C2 * vr, zmm0

        vbroadcastss zmm14,Exp_P1[rax]

        vbroadcastss zmm13,Exp_P2[rax]
        vfmadd213ps zmm8,zmm0,zmm14             # P0 * vf + P1

        vbroadcastss zmm12,Exp_P3[rax]
        vfmadd213ps zmm8,zmm0,zmm13             # * vf + P2

        vbroadcastss zmm15,Exp_P4[rax]
        vfmadd213ps zmm8,zmm0,zmm12             # * vf + P3

        vbroadcastss zmm14,Exp_P5[rax]
        vfmadd213ps zmm8,zmm0,zmm15             # * vf + P4

        vbroadcastss zmm13,Exp_P6[rax]
        vfmadd213ps zmm8,zmm0,zmm14             # * vf + P5

        vbroadcastss zmm12,Exp_X7F[rax]
        vfmadd213ps zmm8,zmm0,zmm13             # * vf + P6

        vcvttps2dq zmm4,zmm4                    # vr to int

        vbroadcastss zmm15,ErfOne[rax]
        vpaddd  zmm4,zmm4,zmm12                 # vr +127

        vpslld  zmm4,zmm4,23

        vmulps  zmm8,zmm8,zmm4                  # exp(ve) = exp(vr) * exp(vf)

        vsubps  zmm8,zmm15,zmm8                 # 1 - exp(ve)

        # merge small numbers' result
        #vorps   zmm8,zmm8,ZMMWORD PTR ErfBuffer1_512[rsp]
        vmovups zmm8{k1}, ZMMWORD PTR BiasGeluBuffer1[rsp]

        # copy sign
        vorps   zmm0,zmm8,ZMMWORD PTR BiasGeluBuffer0[rsp]

.LComputeResidual1x16Loop:

        vfmadd213ps zmm0,zmm20,zmm20 # * temp + temp

        #vmaskmovps ZMMWORD PTR [rsi],zmm3,zmm0  # store the result
        vmovups ZMMWORD PTR [rsi]{k5},zmm0

        add     rdi,16*4
        add     rsi,16*4
        sub     rdx,16
        jg      .LComputeBGBias1x16Loop

.LBiasGeluExit:
        vzeroupper
        add     rsp,BiasGelu_ReturnAddress
        ret

        .end
