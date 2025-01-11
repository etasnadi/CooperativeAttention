# SPDX-FileCopyrightText: Copyright (c) 2025 Ervin Tasnadi <etasnadi@protonmail.com>
# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import ctypes
import math
from time import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np


class ArrayDesc(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("size", ctypes.c_int32)]


class FusedSDPA:

    def __init__(self):
        super().__init__()
        self.lib = ctypes.CDLL((Path(".") / "libsdpa_flash.so").resolve())

        self.lib.callAttention.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_bool,
        ]  # verbosity

        self.lib.initialize.argtypes = [
            ctypes.c_uint32,  # num elems
            ctypes.c_uint32,  # strideB
            ctypes.c_uint32,  # strideH
            ctypes.c_uint32,  # strideS
            ctypes.c_uint32,  # strideD
            ctypes.c_uint32,  # sizeA
            ctypes.c_uint32,  # sizeS
            ctypes.c_uint32,  # sizeC
            ctypes.c_float,  # scaling
            ctypes.c_bool,  # causal
            ctypes.c_bool,
        ]  # verbosity

        self.saved_params = {}

    def call(
        self,
        query,
        key,
        value,
        output,
        intermediate_itemsize,
        scale=None,
        causal=False,
        verbose=False,
        transpose=True,
    ):
        if scale is None:
            scale = math.sqrt(query.size(-1))

        last_parameters = self.saved_params.copy()

        self.saved_params = {
            "strides": (
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query.stride(3),
            ),
            "element_sizes": (
                query.element_size(),
                intermediate_itemsize,
                output.element_size(),
            ),
            "scale": scale,
            "causal": causal,
            "verbose": verbose,
        }

        if last_parameters != self.saved_params:
            self.lib.initialize(
                query.numel(),
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query.stride(3),
                query.element_size(),  # A type (input)
                intermediate_itemsize,  # S type (intermediate)
                output.element_size(),  # C type (accumulation)
                scale,
                causal,
                verbose,
            )

        ptr_q = ctypes.cast(query.cpu().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        if transpose:
            keyt = torch.permute(key.clone(), (0, 1, 3, 2)).contiguous()
            ptr_k = ctypes.cast(keyt.cpu().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        else:
            ptr_k = ctypes.cast(key.cpu().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        ptr_v = ctypes.cast(value.cpu().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        ptr_o = ctypes.cast(output.cpu().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        self.lib.callAttention(ptr_q, ptr_k, ptr_v, ptr_o, verbose)


np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.8f}"})
torch.set_printoptions(profile="full")

torch_src_device = "cpu"
torch_target_device = "cuda"

seed = 101
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# B, H, S, D = 4, 8, 2048, 64
test_dims = [
    (1, 1, 16, 16),
    (1, 1, 32, 16),
    (1, 1, 64, 16),
    (2, 4, 256, 128),
    (4, 8, 2048, 64),
]

test_precisions = [
    ("fp16", "fp16", "fp16"),
    ("fp16", "fp16", "fp32"),
    ("fp16", "fp32", "fp16"),
    ("fp16", "fp32", "fp32"),
]

repeats = 10

test_causal = [True, False]
scale = 1.0

torch_precision = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}

np_precision = {
    "fp16": np.float16,
    "fp32": np.float32,
}

verbose = False

att = FusedSDPA()


def test_call_attention():

    for test_dim in test_dims:
        B, H, S, D = test_dim

        print(
            f"* Setting dimensions to: batch={B}, head={H}, sequence length={S}, embed={D}"
        )

        for precision in test_precisions:

            precision_input = precision[0]
            precision_intermediate = precision[1]
            precision_acc = precision[2]

            print(
                f'* Setting precision to: input="{precision_input}" intermediate="{precision_intermediate}" matrix accumulation="{precision_acc}"'
            )

            for causal in test_causal:

                print(f"* Setting causal to: {causal}")

                for r in range(repeats):

                    # When measuring wall clock, we need to include the memory transfer times for fair comparison
                    query = torch.rand(
                        B,
                        H,
                        S,
                        D,
                        dtype=torch_precision[precision_input],
                        device=torch_src_device,
                    )
                    key = torch.rand(
                        B,
                        H,
                        S,
                        D,
                        dtype=torch_precision[precision_input],
                        device=torch_src_device,
                    )
                    value = torch.rand(
                        B,
                        H,
                        S,
                        D,
                        dtype=torch_precision[precision_input],
                        device=torch_src_device,
                    )

                    # Run torch SDPA
                    torch_start = time()
                    with sdpa_kernel([SDPBackend.MATH]):
                        t_output = F.scaled_dot_product_attention(
                            query.to(torch_target_device),
                            key.to(torch_target_device),
                            value.to(torch_target_device),
                            scale=scale,
                            is_causal=causal,
                        ).to("cpu")
                    torch_elapsed = time() - torch_start

                    # Run vulkan SDPA
                    vk_output = (
                        torch.zeros_like(query, dtype=torch_precision[precision_acc])
                        .cpu()
                        .clone()
                    )

                    vulkan_start = time()
                    att.call(
                        query,
                        key,
                        value,
                        vk_output,
                        torch_precision[precision_intermediate].itemsize,
                        scale,
                        causal,
                        verbose,
                    )
                    vulkan_elapsed = time()

                    mse = float(
                        torch.mean((vk_output.cpu() - t_output) ** 2).cpu().numpy()
                    )
                    print(f"MSE: {mse}")


def main():
    test_call_attention()


if __name__ == "__main__":
    main()
