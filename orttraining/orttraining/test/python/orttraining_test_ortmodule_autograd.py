# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import math
import random
import copy
import torch
from transformers import AutoConfig, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import pytest
from time import sleep
import warnings
from unittest.mock import patch
from collections import OrderedDict
from collections import namedtuple
from inspect import signature

from onnxruntime.training import _utils, ORTModule
import _test_helpers

from torch.nn.parameter import Parameter

import onnx
import torch
torch.manual_seed(1)
from onnxruntime.training import ORTModule
import onnxruntime as ort
import os
from torch.utils.dlpack import from_dlpack, to_dlpack
 
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C
import copy
import numpy as np
import threading
import sys

def _ortvalue_from_dlpack(dlpack_tensor):
    return OrtValue(C.OrtValue.from_dlpack(dlpack_tensor, False))

def run_with_pytorch_on_gpu(model, input_list, output_shape):
    print('Use PyTorch for CUDA run....')
    device = torch.device('cuda:0')
    model.to(device)
    inputs_on_cuda = [input_.to(device) for input_ in input_list]
    output = model(*inputs_on_cuda)
    criterion = torch.nn.MSELoss()

    target=torch.ones(*output_shape).to(device)
    loss = criterion(output, target)
    loss.backward()
    torch.cuda.synchronize()
    return output, [input_.grad for input_ in inputs_on_cuda if input_.requires_grad is True]

def run_with_ort_on_gpu(model, input_list, output_shape):
    print('Use ORTModule for CUDA run....')
    device = torch.device('cuda:0')
    model = copy.deepcopy(model)
    model.to(device)
    model = ORTModule(model)
    inputs_on_cuda = [input_.to(device) for input_ in input_list]
    output = model(*inputs_on_cuda)
    criterion = torch.nn.MSELoss()

    target=torch.ones(*output_shape).to(device)
    loss = criterion(output, target)
    loss.backward()
    torch.cuda.synchronize()
    return output, [input_.grad for input_ in inputs_on_cuda if input_.requires_grad is True]

###################################################################################

@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff*g

class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        print("GeLUFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp


class GeLUModel(torch.nn.Module):
    def __init__(self, output_size):
        super(GeLUModel, self).__init__()
        self.relu = GeLUFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            #self.bias.zero_()
            self.bias.uniform_()

    def forward(self, model_input):
        out = self.relu(model_input, self.bias)
        return out

def compare_numpy_list(val_a, val_b):
    for np_a, np_b in zip(val_a, val_b):
        equal_ = np.allclose(np_a, np_b, 1e-7, 1e-6, equal_nan=True)
        if equal_ is False:
            print("== details ==")
            k=np_a.reshape(-1)[:100]
            l=np_b.reshape(-1)[:100]
            is_equal = np.isclose(k, l, 1e-7, 1e-6, equal_nan=True)
            res = (is_equal + 1) % 2
            diff_indices = np.nonzero(res)
            print(diff_indices)
            print(k, l)
            raise ValueError("find a diff")

    print("outputs matched successfully.")

def test_GeLU():
    output_size = 1024
    m = GeLUModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])
    torch.cuda.synchronize()

    ort.register_forward_core("GeLUFunction", GeLUFunction.apply)
    ort.register_backward_core("GeLUFunction", GeLUFunction.backward)

    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

###################################################################################

# MegatronGFunction is tested in distributed test files
class MegatronFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        print("MegatronFFunction(torch.autograd.Function) forward")
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass the reduce if we are using only 1 GPU.
        return grad_output

class MegatronFModel(torch.nn.Module):
    def __init__(self, output_size):
        super(MegatronFModel, self).__init__()
        self.copy_ = MegatronFFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            #self.bias.zero_()
            self.bias.uniform_()

    def forward(self, model_input):
        model_input = model_input + self.bias
        out = self.copy_(model_input)
        return out

def test_MegatronF():
    output_size = 1024
    m = MegatronFModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])
    torch.cuda.synchronize()

    ort.register_forward_core("MegatronFFunction", MegatronFFunction.apply)
    ort.register_backward_core("MegatronFFunction", MegatronFFunction.backward)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

#############################################################


class ScalarAndTupleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, beta, gamma):
        print("ScalarAndTupleFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.gamma = gamma
        return alpha * beta[0] * beta[1] * gamma * input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        gamma = ctx.gamma
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return alpha * beta[0] * beta[1] * gamma * grad_input, None, None, None

class ScalarAndTupleModel(torch.nn.Module):
    def __init__(self, output_size):
        super(ScalarAndTupleModel, self).__init__()
        self.activation = ScalarAndTupleFunction.apply
        self.linear_a = torch.nn.Linear(output_size, output_size)
        self.linear_b = torch.nn.Linear(output_size, output_size)

    def forward(self, x):
        h = self.linear_a(x)
        h = self.activation(h, 5.0, (-1.0, 2.0), -1.0)
        h = self.linear_b(h)
        return h

def test_ScalarAndTuple():
    output_size = 2
    m = ScalarAndTupleModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])
    torch.cuda.synchronize()

    ort.register_forward_core("ScalarAndTupleFunction", ScalarAndTupleFunction.apply)
    ort.register_backward_core("ScalarAndTupleFunction", ScalarAndTupleFunction.backward)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

#############################################################

class InplaceUpdateInputAsOutputNotRequireGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputAsOutputNotRequireGradFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        return inplace_update_input.add_(3 * bias)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class InplaceUpdateInputAsOutputNotRequireGradModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputAsOutputNotRequireGradModel, self).__init__()
        self.inplace_op = InplaceUpdateInputAsOutputNotRequireGradFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            #self.bias.zero_()
            self.bias.uniform_()

    def forward(self, model_input):
        x = model_input.mul(2)
        y1 = self.inplace_op(self.bias, x) # x did not require grad
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

def test_InplaceUpdateInputAsOutputNotRequireGrad():
    output_size = 1024
    m = InplaceUpdateInputAsOutputNotRequireGradModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])
    torch.cuda.synchronize()

    ort.register_forward_core("InplaceUpdateInputAsOutputNotRequireGradFunction", InplaceUpdateInputAsOutputNotRequireGradFunction.apply)
    ort.register_backward_core("InplaceUpdateInputAsOutputNotRequireGradFunction", InplaceUpdateInputAsOutputNotRequireGradFunction.backward)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

#########################################################################################

class InplaceUpdateInputNotAsOutputNotRequireGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputNotAsOutputNotRequireGradFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        inplace_update_input.add_(3 * bias)
        return inplace_update_input * 5

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class InplaceUpdateInputNotAsOutputNotRequireGradModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputNotAsOutputNotRequireGradModel, self).__init__()
        self.inplace_op = InplaceUpdateInputNotAsOutputNotRequireGradFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            #self.bias.zero_()
            self.bias.uniform_()

    def forward(self, model_input):
        x = model_input.mul(2)
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

def test_InplaceUpdateInputNotAsOutputNotRequireGrad():
    output_size = 1024
    m = InplaceUpdateInputNotAsOutputNotRequireGradModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])
    torch.cuda.synchronize()

    ort.register_forward_core("InplaceUpdateInputNotAsOutputNotRequireGradFunction", InplaceUpdateInputNotAsOutputNotRequireGradFunction.apply)
    ort.register_backward_core("InplaceUpdateInputNotAsOutputNotRequireGradFunction", InplaceUpdateInputNotAsOutputNotRequireGradFunction.backward)

    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

#########################################################################################

class InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        ctx.mark_dirty(inplace_update_input)
        # Be noted: if we make the input dirty, we must also put the input in outputs, otherwise, we will get such an error:
        # "RuntimeError: Some elements marked as dirty during the forward method were not returned as output. The inputs that are modified inplace must all be outputs of the Function.""
        return inplace_update_input.add_(3 * bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass the reduce if we are using only 1 GPU.
        return grad_output, None

class InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyModel, self).__init__()
        self.inplace_op = InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            # self.bias.zero_()
            self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input.mul(2)
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

#                       model_input
#                           |
#                         Mul(2)
#                           |
#                      PythonOP (inplace update)
#                          /   \
#                         /     \
#                      Add       Add
#                        \       /
#                         \     /
#                           Add
#                            |
#                          output0
def test_InplaceUpdateInputAsOutputNotRequireGradWithMarkDirty():
    output_size = 1024
    m = InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()

    ort.register_forward_core("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction", InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction.apply)
    ort.register_backward_core("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction", InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction.backward)

    print("input data: ", x)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    print("comparing forward outputs")
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    print("comparing gradient outputs")
    compare_numpy_list(val_a, val_b)


##########################################################################################
class InplaceUpdateInputAsOutputRequireGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputAsOutputRequireGradFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        # Be noted: if we make the input dirty, we must also put the input in outputs, otherwise, we will get such an error:
        # "RuntimeError: Some elements marked as dirty during the forward method were not returned as output. The inputs that are modified inplace must all be outputs of the Function.""
        return inplace_update_input.add_(3 * bias)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class InplaceUpdateInputAsOutputRequireGradModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputAsOutputRequireGradModel, self).__init__()
        self.inplace_op = InplaceUpdateInputAsOutputRequireGradFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            # self.bias.zero_()
            self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input + self.bias
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

#                       model_input
#                           |
#                         Mul(2)
#                           |
#                      PythonOP (inplace update)
#                          /   \
#                         /     \
#                      Add       Add
#                        \       /
#                         \     /
#                           Add
#                            |
#                          output0
# This case is known to have an warning message: "The output torch tensor @140214094625024, 140212816617984 should reuse the input torch tensor @140214095996104, 140212816617984 but actually not."
# So seems, if we don't have mark_dirty() in auto grad forward, the result is not using the input_, (maybe a view of it, because data address is same)
def test_InplaceUpdateInputAsOutputRequireGrad():
    output_size = 1024
    m = InplaceUpdateInputAsOutputRequireGradModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()

    ort.register_forward_core('InplaceUpdateInputAsOutputRequireGradFunction', InplaceUpdateInputAsOutputRequireGradFunction.apply)
    ort.register_backward_core('InplaceUpdateInputAsOutputRequireGradFunction', InplaceUpdateInputAsOutputRequireGradFunction.backward)

    print("input data: ", x)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    print("comparing forward outputs")
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    print("comparing gradient outputs")
    compare_numpy_list(val_a, val_b)


##########################################################################################

class InplaceUpdateInputNotAsOutputRequireGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputNotAsOutputRequireGradFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        inplace_update_input.add_(3 * bias)
        return inplace_update_input * 5

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class InplaceUpdateInputNotAsOutputRequireGradModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputNotAsOutputRequireGradModel, self).__init__()
        self.inplace_op = InplaceUpdateInputNotAsOutputRequireGradFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            # self.bias.zero_()
            self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input + self.bias
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

#                       model_input
#                           |
#                         Mul(2)
#                           |
#                      PythonOP (inplace update)
#                          /   \
#                         /     \
#                      Add       Add
#                        \       /
#                         \     /
#                           Add
#                            |
#                          output0
# This case is known to have an warning message: "The output torch tensor @140214094625024, 140212816617984 should reuse the input torch tensor @140214095996104, 140212816617984 but actually not."
# So seems, if we don't have mark_dirty() in auto grad forward, the result is not using the input_, (maybe a view of it, because data address is same)
def test_InplaceUpdateInputNotAsOutputRequireGrad():
    output_size = 1024
    m = InplaceUpdateInputNotAsOutputRequireGradModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    ort.register_forward_core('InplaceUpdateInputNotAsOutputRequireGradFunction', InplaceUpdateInputNotAsOutputRequireGradFunction.apply)
    ort.register_backward_core('InplaceUpdateInputNotAsOutputRequireGradFunction', InplaceUpdateInputNotAsOutputRequireGradFunction.backward)

    print("input data: ", x)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    print("comparing forward outputs")
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    print("comparing gradient outputs")
    compare_numpy_list(val_a, val_b)


##########################################################################################

class InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction(torch.autograd.Function) forward, process id {}, thread id {} ====".format(os.getpid(), threading.current_thread().ident))
        ctx.save_for_backward(inplace_update_input, bias)
        ctx.mark_dirty(inplace_update_input)
        # Be noted: if we make the input dirty, we must also put the input in outputs, otherwise, we will get such an error:
        # "RuntimeError: Some elements marked as dirty during the forward method were not returned as output. The inputs that are modified inplace must all be outputs of the Function.""
        return inplace_update_input.add_(3 * bias)

    @staticmethod
    def backward(ctx, grad_output):
        print("InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction(torch.autograd.Function) backward, process id {}, thread id {} ====".format(os.getpid(), threading.current_thread().ident))
        return grad_output, grad_output

class InplaceUpdateInputAsOutputRequireGradWithMarkDirtyModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputAsOutputRequireGradWithMarkDirtyModel, self).__init__()
        self.inplace_op = InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            # self.bias.zero_()
            self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input + self.bias
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

#                       model_input
#                           |
#                         Mul(2)
#                           |
#                      PythonOP (inplace update)
#                          /   \
#                         /     \
#                      Add       Add
#                        \       /
#                         \     /
#                           Add
#                            |
#                          output0
def test_InplaceUpdateInputAsOutputRequireGradWithMarkDirty():
    output_size = 1024
    m = InplaceUpdateInputAsOutputRequireGradWithMarkDirtyModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()

    ort.register_forward_core('InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction', InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction.apply)
    ort.register_backward_core('InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction', InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction.backward)

    print("input data: ", x)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    print("comparing forward outputs")
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    print("comparing gradient outputs")
    compare_numpy_list(val_a, val_b)

def call_python_forward_function(forward_function, requires_grad_flags, tensor_type_flags, *args):
    try:
        wrapped_args = []
        for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args):
            if tensor_flag:
                # Got a tensor. Assume it's a DLPack tensor
                # and convert it to Pytorch tensor.
                wrapped_arg = from_dlpack(arg).detach().clone().contiguous()
                if grad_flag:
                    wrapped_arg.requires_grad = True
                else:
                    wrapped_arg.requires_grad = False

                wrapped_args.append(wrapped_arg)
            else:
                # Use non-tensor as is. It's a PyObject*.
                wrapped_args.append(arg)

        unwrapped_values = []
        ctx = None
        with torch.enable_grad():
            result = forward_function(*wrapped_args)
            if isinstance(result, torch.Tensor):
                # TODO: We need to confirm
                #   1. The ownership of result is transferred to DLPack tensor from Pytorch.
                #   2. The ownership of result is transferred to ORTValue from DLPack.
                # If they are all true, we can remove the object register code below.
                ort_value = _ortvalue_from_dlpack(to_dlpack(result))
                unwrapped_values = [ort_value]
                ctx = result.grad_fn
            elif isinstance(result, tuple) or isinstance(result, list):
                for value in result:
                    unwrapped_value = _ortvalue_from_dlpack(to_dlpack(v))
                    unwrapped_values.append(unwrapped_value)
                    if ctx is not None and hasattr(ctx, 'grad_fn'):
                        ctx = unwrapped_value.grad_fn
            else:
                raise Exception('Unsupported returned type: ', type(result), ' by calling ', forward_function)

        # Must extract one valid context from result tensors.
        assert ctx is not None

        ort.register_python_object(result)
        for value in unwrapped_values:
            # Maintain their life time.
            # This causes memory leak.
            ort.register_python_object(value)

        unwrapped_ptrs = [int(id(ctx))]
        for v in unwrapped_values:
            unwrapped_ptrs.append(int(v.ortvalue_ptr()))

        return tuple(unwrapped_ptrs)
    except:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        sys.stdout.flush()
        sys.stderr.flush()
        raise

def call_python_backward_function(backward_function, requires_grad_flags, tensor_type_flags, *args):
    try:
        wrapped_args = []
        for requires_grad, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args):
            if tensor_flag:
                # Got a tensor. Assume it's a DLPack tensor
                # and convert it to Pytorch tensor.
                wrapped_arg = from_dlpack(arg).clone().contiguous()
                if requires_grad:
                    wrapped_arg.requires_grad = True
                else:
                    wrapped_arg.requires_grad = False
                wrapped_args.append(wrapped_arg)
            else:
                # Use non-tensor as is. It's a PyObject*.
                wrapped_args.append(arg)

        unwrapped_values = []
        result = backward_function(*wrapped_args)
        if isinstance(result, torch.Tensor):
            # TODO: We need to confirm
            #   1. The ownership of result is transferred to DLPack tensor from Pytorch.
            #   2. The ownership of result is transferred to ORTValue from DLPack.
            # If they are all true, we can remove the object register code below.
            ort_value = _ortvalue_from_dlpack(to_dlpack(result))
            unwrapped_values = [ort_value]
        elif isinstance(result, tuple) or isinstance(result, list):
            for value in result:
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise Exception('Unsupported returned element type: ', type(value), ' by calling ', backward_function)
                unwrapped_value = _ortvalue_from_dlpack(to_dlpack(value))
                unwrapped_values.append(unwrapped_value)
        else:
            raise Exception('Unsupported returned type: ', type(result), ' by calling ', backward_function)

        # TODO: release resource at the beginning of each kernel computation.
        ort.register_python_object(result)
        for value in unwrapped_values:
            # Maintain their life time.
            # This causes memory leak.
            ort.register_python_object(value)

        unwrapped_ptrs = []
        for value in unwrapped_values:
            unwrapped_ptrs.append(int(value.ortvalue_ptr()))

        return tuple(unwrapped_ptrs)
    except:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        sys.stdout.flush()
        sys.stderr.flush()
        raise

ort.register_forward_runner(call_python_forward_function)
ort.register_backward_runner(call_python_backward_function)

test_GeLU()
test_MegatronF()
test_ScalarAndTuple()

## test case, some input are in-place updated, and the input did not require gradient.
test_InplaceUpdateInputAsOutputNotRequireGrad()
test_InplaceUpdateInputNotAsOutputNotRequireGrad()
test_InplaceUpdateInputAsOutputNotRequireGradWithMarkDirty()

### test case, some input are in-place updated, and the input require gradient.
test_InplaceUpdateInputAsOutputRequireGrad()
test_InplaceUpdateInputNotAsOutputRequireGrad()
#test_InplaceUpdateInputAsOutputRequireGradWithMarkDirty()
