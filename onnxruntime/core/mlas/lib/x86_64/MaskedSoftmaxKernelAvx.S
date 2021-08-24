/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SoftmaxKernelAvx.s

Abstract:

    This module implements the kernels for the single precision softmax
    operation.

    This implementation uses AVX instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

/*++

Routine Description:

    This routine implements a vectorized kernel to find the maximum value of
    the supplied buffer.

Arguments:

    Input (rdi) - Supplies the input buffer.

    N (rsi) - Supplies the number of elements to process.

Return Value:

    Returns the maximum value of the supplied buffer.

--*/

#        FUNCTION_ENTRY MlasReduceMaximumF32KernelAvx
#        # onnxruntime/onnxruntime/core/mlas/lib/compute.cpp defined this min value, to 8 in ymm0
#        vbroadcastss ymm0,DWORD PTR C_UNDERSCORE(MlasMinimumF32Value)[rip]
#        test    rsi,rsi
#        jz      .LReduceMaximum.ExitKernel
#        cmp     rsi,8
#        jb      .LReduceMaximum.ProcessRemainingCountBy1 # 1 element a time
#        cmp     rsi,32
#        jb      .LReduceMaximum.ProcessRemainingCountBy8 # 1 ymm a time
#        # ymm0-3 hold altogether 4*8 elements
#        vmovaps ymm1,ymm0
#        vmovaps ymm2,ymm0
#        vmovaps ymm3,ymm0
#
## 4 ymm a time
#.LReduceMaximum.ProcessRemainingCountBy32:
#        vmaxps  ymm0,ymm0,YMMWORD PTR [rdi]
#        vmaxps  ymm1,ymm1,YMMWORD PTR [rdi+8*4]
#        sub     rsi,32 # 32 elements, why put this line here?
#        vmaxps  ymm2,ymm2,YMMWORD PTR [rdi+16*4]
#        vmaxps  ymm3,ymm3,YMMWORD PTR [rdi+24*4]
#        add     rdi,32*4                        # advance input by 32 elements
#        cmp     rsi,32
#        jae     .LReduceMaximum.ProcessRemainingCountBy32
#        vmaxps  ymm0,ymm0,ymm1                  # reduce to single vector, min contains in 8 elements
#        vmaxps  ymm2,ymm2,ymm3
#        vmaxps  ymm0,ymm0,ymm2
#
#.LReduceMaximum.ProcessRemainingCountBy8:
#        cmp     rsi,8
#        jb      .LReduceMaximum.ProcessRemainingCountLessThan8
#        vmaxps  ymm0,ymm0,YMMWORD PTR [rdi]
#        sub     rsi,8
#        add     rdi,8*4                         # advance input by 8 elements
#        jmp     .LReduceMaximum.ProcessRemainingCountBy8
#
#.LReduceMaximum.ProcessRemainingCountLessThan8:
#        vextractf128 xmm1,ymm0,1                # reduce to single scalar, 1 means upper 4 elements
#        vmaxps  xmm0,xmm0,xmm1                  # ymm0 lower vs upper
#        vshufps xmm1,xmm0,xmm0,0xEE             # 11 10 11 10 --> x3 x2 x3 x2
#        vmaxps  xmm0,xmm0,xmm1                  # x0 x1 x2 x3 vs x2 x3 x2 x3
#        vshufps xmm1,xmm0,xmm0,0x55             # 01 01 01 01 --> x13 x13 x13 x13
#        vmaxss  xmm0,xmm0,xmm1                  # x02 x13 ... ... vs x13 x13 x13 x13, vmaxss means just pick the first scalar float
#        test    rsi,rsi
#        jz      .LReduceMaximum.ExitKernel
#
#.LReduceMaximum.ProcessRemainingCountBy1:       # compare xmm to the remaining
#        vmaxss  xmm0,xmm0,DWORD PTR [rdi]
#        add     rdi,4                           # advance input by 1 element
#        dec     esi
#        jnz     .LReduceMaximum.ProcessRemainingCountBy1
#
#// first element of xmm0 is the return value
#.LReduceMaximum.ExitKernel:
#        vzeroupper
#        ret

/*++

Routine Description:

    This routine implements a vectorized kernel to produce the final output for
    the softmax operation.

Arguments:

    Output (rdi) - Supplies the output buffer.

    N (rsi) - Supplies the number of elements to process.

    Parameters (rdx) - Supplies an array containing the scale value.

    N1 (rcx) - masked length, N1 >= N

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasComputeSoftmaxOutputF32KernelAvx

        vbroadcastss ymm4,DWORD PTR [rdx]       # broadcast scale value to 8 elements
        cmp     rsi,32                          # 8 * 4
        jb      .LComputeSoftmaxOutput.ProcessRemainingCountBy8

.LComputeSoftmaxOutput.ProcessRemainingCountBy32:
        vmulps  ymm0,ymm4,YMMWORD PTR [rdi]     # in8
        vmulps  ymm1,ymm4,YMMWORD PTR [rdi+8*4] # in16
        sub     rsi,32                          # should not be here, processed 32 elements
        vmulps  ymm2,ymm4,YMMWORD PTR [rdi+16*4]# in24
        vmulps  ymm3,ymm4,YMMWORD PTR [rdi+24*4]# in32
        vmovups YMMWORD PTR [rdi],ymm0          # output
        vmovups YMMWORD PTR [rdi+8*4],ymm1
        vmovups YMMWORD PTR [rdi+16*4],ymm2
        vmovups YMMWORD PTR [rdi+24*4],ymm3
        add     rdi,32*4                        # advance output by 32 elements
        cmp     rsi,32
        jae     .LComputeSoftmaxOutput.ProcessRemainingCountBy32

.LComputeSoftmaxOutput.ProcessRemainingCountBy8:
        cmp     rsi,8
        jb      .LComputeSoftmaxOutput.ProcessRemainingCountLessThan8
        vmulps  ymm0,ymm4,YMMWORD PTR [rdi]
        sub     rsi,8
        vmovups YMMWORD PTR [rdi],ymm0
        add     rdi,8*4                         # advance output by 8 elements
        jmp     .LComputeSoftmaxOutput.ProcessRemainingCountBy8

.LComputeSoftmaxOutput.ProcessRemainingCountLessThan8:
        test    rsi,rsi
        jz      .LComputeSoftmaxOutput.ExitKernel

.LComputeSoftmaxOutput.ProcessRemainingCountBy1:
        vmulss  xmm0,xmm4,DWORD PTR [rdi]
        vmovss  DWORD PTR [rdi],xmm0
        add     rdi,4                           # advance output by 1 element
        dec     esi
        jnz     .LComputeSoftmaxOutput.ProcessRemainingCountBy1

.LComputeSoftmaxOutput.ExitKernel:
        vzeroupper
        ret


/*++

Routine Description:

    This routine implements a vectorized kernel for the sum of exponential
    functions.

Arguments:

    Input (rdi) - Supplies the input buffer.

    Output (rsi) - Optionally supplies the output buffer. When used for Softmax,
        the output buffer is used to store the intermediate exp() results. When
        used for LogSoftmax, the intermediate exp() results are not required.

    N (rdx) - Supplies the number of elements to process.

    NegativeMaximum (rcx) - Supplies the address of the negative maximum that
        is added to each element before computing the exponential function.

Return Value:

    Returns the sum of the exponential functions.

--*/

#        FUNCTION_ENTRY MlasComputeSumExpF32KernelAvx512F
#
#        lea     rax,C_UNDERSCORE(MlasExpConstants)[rip]
#        vbroadcastss zmm21,.LExpConstants_LowerRange[rax] # -103.9720840454f
#        vbroadcastss zmm22,.LExpConstants_RoundingBias[rax] # 12582912.f
#        vbroadcastss zmm23,.LExpConstants_Log2Reciprocal[rax] # 1.44269504088896341f 1/ln2
#        vbroadcastss zmm24,.LExpConstants_Log2High[rax] # -6.93145752e-1f, e^ = 0.5
#        vbroadcastss zmm25,.LExpConstants_Log2Low[rax] #  -1.42860677e-6f, e^ = 1
#        vbroadcastss zmm26,.LExpConstants_poly_0[rax] # 0x1.694000p-10, 0.00137806
#        vbroadcastss zmm27,.LExpConstants_poly_1[rax] # 0x1.125edcp-7, 0.00837312
#        vbroadcastss zmm28,.LExpConstants_poly_2[rax] # 0x1.555b5ap-5, 0.0416695
#        vbroadcastss zmm29,.LExpConstants_poly_3[rax] # 0x1.555450p-3, 0.166665
#        vbroadcastss zmm30,.LExpConstants_poly_4[rax] # 0x1.fffff6p-2, 0.5
#        vbroadcastss zmm31,.LExpConstants_poly_56[rax] # 0x1.000000p+0, 1
#
#        vbroadcastss zmm19,DWORD PTR [rcx]      # broadcast negative maximum value zmm19: negmin
#        vpxord  zmm20,zmm20,zmm20               # clear exp() accumulator
#        sub     rdx,48                          # D -=48
#        jb      .LComputeSumExp.ProcessRemainingCount
#
#.LComputeSumExp.ComputeExpBy48Loop:
#        vaddps  zmm0,zmm19,ZMMWORD PTR [rdi]    # bias by negative maximum value, zmm0 in16
#        vaddps  zmm3,zmm19,ZMMWORD PTR [rdi+64] # zmm3 in32
#        vaddps  zmm16,zmm19,ZMMWORD PTR [rdi+128] # zmm16 in48
#        vmaxps  zmm0,zmm21,zmm0                 # clamp lower bound for cin16
#        vmovaps zmm2,zmm23                      # zmm2 = zmm23 = 1/ln2
#        vmaxps  zmm3,zmm21,zmm3                 # clamp lower bound for cin32
#        vmovaps zmm5,zmm23                      # zmm5 = zmm23 = 1/ln2
#        vmaxps  zmm16,zmm21,zmm16               # clamp lower bound for cin48
#        vmovaps zmm18,zmm23                     # zmm18 = zmm23 = 1/ln2
#        vfmadd213ps zmm2,zmm0,zmm22             # (input / ln2) plus rounding bias, zmm2:cin16lb, zmm5: cin32lb, zmm18: cin48lb
#        vfmadd213ps zmm5,zmm3,zmm22
#        vfmadd213ps zmm18,zmm16,zmm22
#        vmovaps zmm1,zmm26                      # p = poly_0
#        vmovaps zmm4,zmm26
#        vmovaps zmm17,zmm26
#        vsubps  zmm2,zmm2,zmm22                 # m = round(input / ln2), cin16lbr, cin32lbr, cin48lbr, closest int
#        vsubps  zmm5,zmm5,zmm22
#        vsubps  zmm18,zmm18,zmm22
#        vfmadd231ps zmm0,zmm2,zmm24             # range reduce: x -= (m * ln2_high), cin16 -= cin16lbr * (ln2_high + ln2_low)
#        vfmadd231ps zmm3,zmm5,zmm24
#        vfmadd231ps zmm16,zmm18,zmm24
#        vfmadd231ps zmm0,zmm2,zmm25             # range reduce: x -= (m * ln2_low), remaining is the decimal part
#        vfmadd231ps zmm3,zmm5,zmm25
#        vfmadd231ps zmm16,zmm18,zmm25
#        vfmadd213ps zmm1,zmm0,zmm27             # p = p * x + poly_1, accum zmm1|4|17 as p on cin
#        vfmadd213ps zmm4,zmm3,zmm27
#        vfmadd213ps zmm17,zmm16,zmm27
#        vfmadd213ps zmm1,zmm0,zmm28             # p = p * x + poly_2
#        vfmadd213ps zmm4,zmm3,zmm28
#        vfmadd213ps zmm17,zmm16,zmm28
#        vfmadd213ps zmm1,zmm0,zmm29             # p = p * x + poly_3
#        vfmadd213ps zmm4,zmm3,zmm29
#        vfmadd213ps zmm17,zmm16,zmm29
#        vfmadd213ps zmm1,zmm0,zmm30             # p = p * x + poly_4
#        vfmadd213ps zmm4,zmm3,zmm30
#        vfmadd213ps zmm17,zmm16,zmm30
#        vfmadd213ps zmm1,zmm0,zmm31             # p = p * x + poly_5
#        vfmadd213ps zmm4,zmm3,zmm31
#        vfmadd213ps zmm17,zmm16,zmm31
#        vfmadd213ps zmm1,zmm0,zmm31             # p = p * x + poly_6
#        vfmadd213ps zmm4,zmm3,zmm31
#        vfmadd213ps zmm17,zmm16,zmm31
#        vscalefps zmm1,zmm1,zmm2                # p16 *= 2^cin16lbrf
#        vscalefps zmm4,zmm4,zmm5
#        vscalefps zmm17,zmm17,zmm18
#        vaddps  zmm20,zmm20,zmm1                # accumulate exp() results
#        vaddps  zmm20,zmm20,zmm4
#        vaddps  zmm20,zmm20,zmm17
#        add     rdi,48*4                        # advance input by 48 elements
#        test    rsi,rsi                         # nullptr outptut, no need store
#        jz      .LComputeSumExp.SkipStoreResultsBy48
#        vmovups ZMMWORD PTR [rsi],zmm1          # output p, also accumlated the sum on zmm20
#        vmovups ZMMWORD PTR [rsi+64],zmm4
#        vmovups ZMMWORD PTR [rsi+128],zmm17
#        add     rsi,48*4                        # advance output by 48 elements
#
#.LComputeSumExp.SkipStoreResultsBy48:
#        sub     rdx,48                          # D -= 48
#        jae     .LComputeSumExp.ComputeExpBy48Loop # loop
#
#.LComputeSumExp.ProcessRemainingCount:
#        add     rdx,48                          # correct for over-subtract above
#        jz      .LComputeSumExp.ReduceAccumulator # there is no remaining, directly go to result
#        mov     eax,-1
#        kmovw   k1,eax                          # update mask to access all elements, suppose 16 elements at least for now
#
#.LComputeSumExp.ComputeExpBy16Loop:
#        cmp     rdx,16
#        jae     .LComputeSumExp.ProcessSingleVector
#        # platform.cpp
#        # 0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F,
#        # 0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF,
#        lea     r10,C_UNDERSCORE(MlasOpmask16BitTableAvx512)[rip]
#        kmovw   k1,WORD PTR [r10+rdx*2] # rdx * 2byte
#
#.LComputeSumExp.ProcessSingleVector:
#        vaddps  zmm0{k1}{z},zmm19,ZMMWORD PTR [rdi]
#                                                # bias by negative maximum value
#        vmaxps  zmm0,zmm21,zmm0                 # clamp lower bound
#        vmovaps zmm2,zmm23                      # 1/ln2
#        vfmadd213ps zmm2,zmm0,zmm22             # (input / ln2) plus rounding bias
#        vmovaps zmm1,zmm26                      # p = poly_0
#        vsubps  zmm2,zmm2,zmm22                 # m = round(input / ln2)
#        vfmadd231ps zmm0,zmm2,zmm24             # range reduce: x -= (m * ln2_high)
#        vfmadd231ps zmm0,zmm2,zmm25             # range reduce: x -= (m * ln2_low)
#        vfmadd213ps zmm1,zmm0,zmm27             # p = p * x + poly_1
#        vfmadd213ps zmm1,zmm0,zmm28             # p = p * x + poly_2
#        vfmadd213ps zmm1,zmm0,zmm29             # p = p * x + poly_3
#        vfmadd213ps zmm1,zmm0,zmm30             # p = p * x + poly_4
#        vfmadd213ps zmm1,zmm0,zmm31             # p = p * x + poly_5
#        vfmadd213ps zmm1,zmm0,zmm31             # p = p * x + poly_6
#        vscalefps zmm1,zmm1,zmm2                # p *= 2^floor m
#        vaddps  zmm20{k1},zmm20,zmm1            # accumulate exp() results
#        add     rdi,16*4                        # advance input by 16 elements
#        test    rsi,rsi
#        jz      .LComputeSumExp.SkipStoreResultsBy16
#        vmovups ZMMWORD PTR [rsi]{k1},zmm1      # put to result if output not null
#        add     rsi,16*4                        # advance output by 16 elements
#
#.LComputeSumExp.SkipStoreResultsBy16:
#        sub     rdx,16
#        ja      .LComputeSumExp.ComputeExpBy16Loop
#
#.LComputeSumExp.ReduceAccumulator:
#        vextractf64x4 ymm0,zmm20,1              # higher 8 elements to ymm0
#        vaddps  zmm0,zmm0,zmm20                 # add them all to ymm0
#        vhaddps ymm0,ymm0,ymm0                  # y01 y23 y01 y23 y45 y67 y45 y67
#        vhaddps ymm0,ymm0,ymm0                  # y0123 y0123 0123 y0123 y4567 ...
#        vextractf128 xmm1,ymm0,1                # y4567 .... > xmm1
#        vaddss  xmm0,xmm0,xmm1                  # y01234567
#
#        vzeroupper                              # upper 256 bits of 0~15
#        ret
#
#        .end