/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelBiasAvx512FCommon.h

Abstract:

    This module implements the kernels for the floating point matrix/matrix
    multiply operation (SGEMM and DGEMM).

    This implementation uses AVX512F instructions.

--*/
// no need following line since in SgemmKernelAvx512F.S it is included above this file
// #include "FgemmKernelAvx512FCommon.h"

        .equ    .LFgemmKernelFrame_Bias, 56
        .equ    .LFgemmKernelFrame_R8, 64

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

Implicit Arguments:
    r8 is the only one unused, rbp is for temp, rsp is emmmm

    rdi - Supplies the address of matrix A.
    rbx - Supplies the address of matrix A + 3.
    r13 - Supplies the address of matrix A + 6.
    r14 - Supplies the address of matrix A + 9.

    rsi - Supplies the address of matrix B.

    r11 - Supplies the address of matrix A.

    r12 - 64 * CountK

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    rdx - Supplies the address of matrix C.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    rax - Supplies the length in bytes of a row from matrix C.

    r15 - Stores the ZeroMode argument from the stack frame.

--*/

        .macro ProcessCountMBias RowCount

        mov     .LFgemmKernelFrame_R8[rsp],r8 # rbp + 48
        mov     r8,.LFgemmKernelFrame_Bias[rsp] # rbp + 40

        cmp     r9,.LFgemmZmmElementCount       # 16 elements, 512 bits
        jbe     .LProcessRemainingCountNBias\@

.LProcessNextColumnLoop2xNBias\@:
        # 6 x 2, this is because vzeroall not affecting zmm16 above
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm16,zmm4"
                                            # clear upper block accumulators
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm17,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm18,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm19,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm20,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm21,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm22,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm23,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm24,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm25,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm26,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm27,zmm5"
        # ComputeBlockLoop is in Common.h
        # assgining A(rdi, rbx, r13 and r14), B(rsi), # reload A and advance/store C is at the end of this 2 block column calculation
        # we get 12 x 2 of C calculated at this time, and they are at zmm4~zmm27 waiting for alpha and bias
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy2, \RowCount\()
        add     rsi,r12                     # advance matrix B by 64*CountK bytes, since we handled one more column during above func
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha2xNBlockBias\@
        # aAB + C for 12 x 1 first column
        # why bother use vfmadd231 when alpha is 1.0?
        EmitIfCountGE \RowCount\(), 1, "vfmadd213pf zmm4,zmm31,ZMMWORD PTR [r8]"       # orig: [rdx] C
        EmitIfCountGE \RowCount\(), 2, "vfmadd213pf zmm6,zmm31,ZMMWORD PTR [r8]"       # orig: [rdx+rax] C + ldc
        EmitIfCountGE \RowCount\(), 3, "vfmadd213pf zmm8,zmm31,ZMMWORD PTR [r8]"       # orig: [rdx+rax*2] C + 2ldc
        EmitIfCountGE \RowCount\(), 4, "vfmadd213pf zmm10,zmm31,ZMMWORD PTR [r8]"      # orig: [rbx] C + 3ldc
        EmitIfCountGE \RowCount\(), 5, "vfmadd213pf zmm12,zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213pf zmm14,zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm16,zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm18,zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm20,zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm22,zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm24,zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm26,zmm31,ZMMWORD PTR [r8]"
        add     r8,.LFgemmElementSize
        jmp     .LStore2xNBlockBias\@

# Zero mode, only need to multiply alpha, no bias
.LMultiplyAlpha2xNBlockBias\@:
        EmitIfCountGE \RowCount\(), 1, "vmulpf zmm4,zmm4,zmm31"
        EmitIfCountGE \RowCount\(), 2, "vmulpf zmm6,zmm6,zmm31"
        EmitIfCountGE \RowCount\(), 3, "vmulpf zmm8,zmm8,zmm31"
        EmitIfCountGE \RowCount\(), 4, "vmulpf zmm10,zmm10,zmm31"
        EmitIfCountGE \RowCount\(), 5, "vmulpf zmm12,zmm12,zmm31"
        EmitIfCountGE \RowCount\(), 6, "vmulpf zmm14,zmm14,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm16,zmm16,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm18,zmm18,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm20,zmm20,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm22,zmm22,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm24,zmm24,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm26,zmm26,zmm31"

# destination might be unaligned
.LStore2xNBlockBias\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf ZMMWORD PTR [rdx],zmm4"
        EmitIfCountGE \RowCount\(), 2, "vmovupf ZMMWORD PTR [rdx+rax],zmm6"
        EmitIfCountGE \RowCount\(), 3, "vmovupf ZMMWORD PTR [rdx+rax*2],zmm8"
        EmitIfCountGE \RowCount\(), 4, "vmovupf ZMMWORD PTR [rbx],zmm10"
        EmitIfCountGE \RowCount\(), 5, "vmovupf ZMMWORD PTR [rbx+rax],zmm12"
        EmitIfCountGE \RowCount\(), 6, "vmovupf ZMMWORD PTR [rbx+rax*2],zmm14"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13],zmm16"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13+rax],zmm18"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13+rax*2],zmm20"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14],zmm22"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14+rax],zmm24"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14+rax*2],zmm26"
        add     rdx,64                      # advance matrix C by ZMMWORD
.if \RowCount\() > 3
        add     rbx,64                      # advance matrix C plus 3 rows by ZMMWORD
.if \RowCount\() == 12
        add     r13,64                      # advance matrix C plus 6 rows by ZMMWORD
        add     r14,64                      # advance matrix C plus 9 rows by ZMMWORD
.endif
.endif
        sub     r9,.LFgemmZmmElementCount   # CountN -= 16

.LOutput1xNBlockBias\@:
        sub     r9,.LFgemmZmmElementCount   # CountN -= 16
        jae     .LOutput1xNBlockWithMaskBias\@
        # mask preparation for k1
        lea     rcx,[r9+.LFgemmZmmElementCount]
                                            # correct for over-subtract above
        mov     ebp,1
        shl     ebp,cl
        dec     ebp
        kmovw   k1,ebp                      # update mask for remaining columns
        xor     r9,r9                       # no more columns remaining

.LOutput1xNBlockWithMaskBias\@:
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha1xNBlockWithMaskBias\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213pf zmm5{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213pf zmm7{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213pf zmm9{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213pf zmm11{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213pf zmm13{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213pf zmm15{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm17{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm19{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm21{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm23{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm25{k1},zmm31,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm27{k1},zmm31,ZMMWORD PTR [r8]"
        add     r8,.LFgemmElementSize
        jmp     .LStore1xNBlockWithMaskBias\@

.LMultiplyAlpha1xNBlockWithMaskBias\@:
        EmitIfCountGE \RowCount\(), 1, "vmulpf zmm5,zmm5,zmm31"
        EmitIfCountGE \RowCount\(), 2, "vmulpf zmm7,zmm7,zmm31"
        EmitIfCountGE \RowCount\(), 3, "vmulpf zmm9,zmm9,zmm31"
        EmitIfCountGE \RowCount\(), 4, "vmulpf zmm11,zmm11,zmm31"
        EmitIfCountGE \RowCount\(), 5, "vmulpf zmm13,zmm13,zmm31"
        EmitIfCountGE \RowCount\(), 6, "vmulpf zmm15,zmm15,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm17,zmm17,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm19,zmm19,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm21,zmm21,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm23,zmm23,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm25,zmm25,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm27,zmm27,zmm31"

.LStore1xNBlockWithMaskBias\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf ZMMWORD PTR [rdx]{k1},zmm5"
        EmitIfCountGE \RowCount\(), 2, "vmovupf ZMMWORD PTR [rdx+rax]{k1},zmm7"
        EmitIfCountGE \RowCount\(), 3, "vmovupf ZMMWORD PTR [rdx+rax*2]{k1},zmm9"
        EmitIfCountGE \RowCount\(), 4, "vmovupf ZMMWORD PTR [rbx]{k1},zmm11"
        EmitIfCountGE \RowCount\(), 5, "vmovupf ZMMWORD PTR [rbx+rax]{k1},zmm13"
        EmitIfCountGE \RowCount\(), 6, "vmovupf ZMMWORD PTR [rbx+rax*2]{k1},zmm15"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13]{k1},zmm17"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13+rax]{k1},zmm19"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13+rax*2]{k1},zmm21"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14]{k1},zmm23"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14+rax]{k1},zmm25"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14+rax*2]{k1},zmm27"
        add     rdx,64                      # advance matrix C by ZMMWORD
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,.LFgemmZmmElementCount   # whether remaining columns(CountN) is greater than 16
        ja      .LProcessNextColumnLoop2xNBias\@
        test    r9,r9 # if no more columns(CountN) just exit

        mov     r8,.LFgemmKernelFrame_R8[rsp]# rbp + 48
        # mov     .LFgemmKernelFrame_Bias[rsp],r8 # rbp + 40

        jz      .LExitKernelBias

.LProcessRemainingCountNBias\@:
        # just care for the first column
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm17,zmm5"
                                            # clear upper block accumulators
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm19,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm21,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm23,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm25,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm27,zmm5"
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy1, \RowCount\()
        jmp     .LOutput1xNBlockBias\@

        .endm

/*++

Macro Description:

    This macro generates the inner kernel to compute matrix multiplication.

Arguments:

    FunctionName - Supplies the name for the generated function.

--*/

        .macro FgemmKernelBiasAvx512FFunction FunctionName

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (rdi) - Supplies the address of matrix A.

    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C (rdx) - Supplies the address of matrix C.

    CountK (rcx) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A. rbp + 16

    ldc - Supplies the first dimension of matrix C. rbp + 24

    Alpha (xmm0) - Supplies the scalar alpha multiplier (see GEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into. rbp + 32

    Bias - Supplies the address of Bias vector for the last dimension rbp + 40

Return Value:

    Returns the number of rows handled.

--*/

        FUNCTION_ENTRY \FunctionName\()

        push    rbp                         # rbp
        push    rbx                         # rbp - 8
        push    r15                         # rbp - 16
        mov     .LFgemmKernelFrame_SavedR12[rsp],r12 # rbp - 56
        mov     .LFgemmKernelFrame_SavedR13[rsp],r13 # rbp - 48
        mov     .LFgemmKernelFrame_SavedR14[rsp],r14 # rbp - 40
        mov     r11,rdi                     # A
        mov     r10,.LFgemmKernelFrame_lda[rsp] # lda, rbp + 16
        shl     r10,.LFgemmElementShift     # convert lda to bytes
        mov     rax,.LFgemmKernelFrame_ldc[rsp] # ldc, rbp + 24
        shl     rax,.LFgemmElementShift     # convert ldc to bytes
        mov     r12,rcx                     # CountK
        shl     r12,6                       # compute 64*CountK bytes
        mov     ebp,-1
        kmovw   k1,ebp                      # 0xFFFF, update mask to write all columns
        movzx   r15,BYTE PTR .LFgemmKernelFrame_ZeroMode[rsp] # rbp + 32
        vbroadcastsf zmm31,xmm0 # zmm31: [Alpha] x 16
        vzeroall # zmm16-zmm31 are not affected

//
// Process CountM rows of the matrices.
//

        cmp     r8,12                       # CountM
        jb      .LProcessCountMLessThan12Bias
        mov     r8d,12                      # r8d 32 bits will go to eax, return 12 rows handled
        ProcessCountMBias 12

.LProcessCountMLessThan12Bias:
        cmp     r8,5
        ja      .LProcessCountM6Bias
        je      .LProcessCountM5Bias
        cmp     r8,3
        ja      .LProcessCountM4Bias
        je      .LProcessCountM3Bias
        cmp     r8,1
        je      .LProcessCountM1Bias

.LProcessCountM2Bias:
        ProcessCountMBias 2

.LProcessCountM4Bias:
        ProcessCountMBias 4

.LProcessCountM6Bias:
        mov     r8d,6                       # return 6 rows handled
        ProcessCountMBias 6

//
// Restore non-volatile registers and return.
//

.LExitKernelBias:
        mov     eax,r8d # return value
        mov     r12,.LFgemmKernelFrame_SavedR12[rsp] # restore r12
        mov     r13,.LFgemmKernelFrame_SavedR13[rsp] # restore r13
        mov     r14,.LFgemmKernelFrame_SavedR14[rsp] # restore r14
        pop     r15     # restore r15, and rsp by 8
        pop     rbx     # restore rbx, and rsp by 8
        pop     rbp     # restore rbp, and rsp by 8, rsp is restored at this point
        ret

.LProcessCountM1Bias:
        ProcessCountMBias 1

.LProcessCountM3Bias:
        ProcessCountMBias 3

.LProcessCountM5Bias:
        ProcessCountMBias 5

        .endm
