/*
SPDX-FileCopyrightText: 2025 Ervin Tasnadi <etasnadi@protonmail.com>
SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

/*

Computes the scaled dot product attention (SDPA) using Vulkan with the
cooperative matrix extension for accelerated matrix multiplication.

The interface is similar that of Pytorch's
torch.nn.functional.scaled_dot_product_attention.

Types for mixed precision computation:
Template parameter name (kernel parameter name)
* InputType (A type): matrix multiplication input type
* IntermediateType (S type): computation type for scaling and normalization
* AccumulatorType (C type): matrix multiplication accummulator type

Input shape: B, H, S, D, where
* B is the batch size,
* H is the number of heads,
* S is the context length
* D is the embedding dimension

Each input is row-major. K is assumed to be transposed.

*/

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <stdfloat>
#include <string.h>
#include <vector>

#include <vulkan/vulkan.h>

using std::vector;

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))

#define CHECK_RESULT(r)                                                        \
  do {                                                                         \
    if ((r) != VK_SUCCESS) {                                                   \
      throw std::invalid_argument(                                             \
          std::string("Vulkan error. Code: ") + std::to_string(r) +            \
          ", file: " +                                                         \
          std::filesystem::path(std::string(__FILE__)).filename().string() +   \
          ", line: " + std::to_string(__LINE__));                              \
    }                                                                          \
  } while (0)

#define FMT_BEG "\033["
#define FMT_CLR "\033[0m"

#define MATRIX_OP_OVERWRITE 0
#define MATRIX_OP_ADD 1

#define COPY_HOST_TO_DEVICE 0
#define COPY_DEVICE_TO_HOST 1

class VulkanContext;

namespace VulkanHelper {
// Vulkan related utility functions

std::vector<std::string> componentTypeNames = {
    "float16_t", "float32_t", "float64_t", "int8_t",   "int16_t", "int32_t",
    "int64_t",   "uint8_t",   "uint16_t",  "uint32_t", "uint64_t"};

std::vector<std::string> shortComponentTypeNames = {
    "fp16", "fp32", "fp64", "s8",  "s16", "s32",
    "s64",  "u8",   "u16",  "u32", "u64"};

std::vector<uint32_t> componentTypeSizes = {16, 32, 64, 8,  16, 32,
                                            64, 8,  16, 32, 64};

vector<char> loadSPIRV(std::string fileName) {
  std::ifstream spirvfile(fileName.c_str(), std::ios::binary | std::ios::ate);
  std::streampos spirvsize = spirvfile.tellg();
  if ((int)spirvsize == -1) {
    throw std::invalid_argument(std::string("File not found: ") + fileName);
  }
  spirvfile.seekg(0, std::ios::beg);

  vector<char> spirv(spirvsize);
  spirvfile.read(&spirv[0], spirvsize);
  return spirv;
}

VkCommandBufferBeginInfo getCommandBufferBeginInfo() {
  VkCommandBufferBeginInfo info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      NULL,
      0,
      NULL,
  };

  return info;
}

VkSubmitInfo
getSingleCommandSubmitInfo(const std::vector<VkCommandBuffer> &commandBuffers) {
  VkSubmitInfo info = {
      VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0,    NULL, NULL, 1,
      commandBuffers.data(),         0,    NULL,
  };

  return info;
}

// pasted from Vulkan spec
int32_t
findProperties(const VkPhysicalDeviceMemoryProperties *pMemoryProperties,
               uint32_t memoryTypeBitsRequirement,
               VkMemoryPropertyFlags requiredProperties) {
  const uint32_t memoryCount = pMemoryProperties->memoryTypeCount;
  for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
    const uint32_t memoryTypeBits = (1 << memoryIndex);
    const bool isRequiredMemoryType =
        memoryTypeBitsRequirement & memoryTypeBits;

    const VkMemoryPropertyFlags properties =
        pMemoryProperties->memoryTypes[memoryIndex].propertyFlags;
    const bool hasRequiredProperties =
        (properties & requiredProperties) == requiredProperties;

    if (isRequiredMemoryType && hasRequiredProperties)
      return static_cast<int32_t>(memoryIndex);
  }

  // failed to find memory type
  return -1;
}

void createArray(VkDevice device,
                 VkPhysicalDeviceMemoryProperties memoryProperties,
                 VkDeviceSize bufferSize, VkBuffer &buffer,
                 VkDeviceMemory &memory, uint32_t memorySettings,
                 VkBufferUsageFlags bufferUsageFlags) {

  VkBufferCreateInfo bufferCreateInfo = {
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      NULL,
      0,
      bufferSize,
      bufferUsageFlags,
      VK_SHARING_MODE_EXCLUSIVE,
      0u,
      NULL,
  };

  CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer));

  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device, buffer, &memReqs);

  int32_t memoryIndex =
      findProperties(&memoryProperties, memReqs.memoryTypeBits, memorySettings);

  VkMemoryAllocateFlagsInfo memAllocateFlagsInfo = {
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      NULL,
      VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, // The memory can back a buffer
                                             // that can be accessed from a
                                             // shader AND its pointer can be
                                             // queried using
                                             // vkGetDeviceMemoryOpaqueCaptureAddress
      0,
  };

  VkMemoryAllocateInfo memAllocateInfo = {
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      &memAllocateFlagsInfo,
      memReqs.size,
      (uint32_t)memoryIndex,
  };

  CHECK_RESULT(vkAllocateMemory(device, &memAllocateInfo, NULL, &memory));
  CHECK_RESULT(vkBindBufferMemory(device, buffer, memory, 0));
}

template <typename T> VkComponentTypeKHR getVulkanType() {
  static_assert(false,
                "Vulkan type is not defined for the requested C++ type.");
  return 0;
}

uint32_t getTypeSize(VkComponentTypeKHR componentType) {
  return componentTypeSizes[componentType] / 8;
}

std::string getTypeName(VkComponentTypeKHR componentType) {
  return componentTypeNames[componentType];
}

std::string getShortTypeName(VkComponentTypeKHR componentType) {
  return shortComponentTypeNames[componentType];
}

template <> VkComponentTypeKHR getVulkanType<float>() {
  return VK_COMPONENT_TYPE_FLOAT32_KHR;
}

template <> VkComponentTypeKHR getVulkanType<std::float16_t>() {
  return VK_COMPONENT_TYPE_FLOAT16_KHR;
}

} // namespace VulkanHelper

class Array {
public:
  virtual void *const getPtr() const = 0;
  virtual size_t getSize() const = 0;
  virtual void copyDataFrom(Array &src) const = 0;
  ~Array(){};
};

// Manages a memory buffer that is either owned by the object or not.
class HostArray : public Array {
private:
  void *const ptr = nullptr;
  size_t bufferSize;
  bool bufferBorrowed = false;

public:
  HostArray() = delete;

  HostArray(const HostArray &) = delete;

  HostArray(void *aPtr, size_t aBufferSize)
      : ptr(aPtr), bufferSize(aBufferSize) {
    bufferBorrowed = true;
  }

  HostArray(size_t aBufferSize)
      : ptr((void *)new char[aBufferSize]), bufferSize(aBufferSize) {}

  void *const getPtr() const override { return ptr; }

  void copyDataFrom(Array &src) const override {
    std::memcpy(ptr, src.getPtr(), std::min(bufferSize, src.getSize()));
  }

  size_t getSize() const override { return bufferSize; }

  ~HostArray() {
    if (!bufferBorrowed) {
      delete static_cast<char *>(ptr);
    }
  }
};

/*
Does matrix ops on an externally provided or a fully owned buffer.
*/
template <typename T> class MatrixDesc {
private:
  std::shared_ptr<Array> buffer;
  uint32_t offset = 0;
  T *getPtr() const { return ((T *)buffer->getPtr()) + offset / sizeof(T); }

public:
  struct {
    uint32_t rows = 0, cols = 0;
  } dims;

  MatrixDesc() {}

  MatrixDesc(int rows, int cols, std::shared_ptr<Array> aBuffer,
             uint32_t aOffset = 0) {
    dims.rows = rows;
    dims.cols = cols;
    if (aOffset + rows * cols * sizeof(T) > aBuffer->getSize()) {
      throw std::invalid_argument(
          std::string("The matrix does not fit into the buffer:") +
          std::string(" offset=") + std::to_string(aOffset) +
          std::string(" rows*cols=") + std::to_string(rows * cols) +
          std::string(" sizeof(T)=") + std::to_string(sizeof(T)) +
          std::string(" bufferSize=") + std::to_string(aBuffer->getSize()));
    }
    offset = aOffset;
    buffer = aBuffer;
  }

  MatrixDesc(int rows, int cols) {
    dims.rows = rows;
    dims.cols = cols;
    buffer = std::make_shared<HostArray>(rows * cols * sizeof(T));
  }

  MatrixDesc(int rows, int cols, T value) {
    dims.rows = rows;
    dims.cols = cols;
    buffer = std::make_shared<HostArray>(rows * cols * sizeof(T));
    for (int i = 0; i < rows * cols; i++) {
      setDataFloat(i, value);
    }
  }

  MatrixDesc extractSubMatrix(int row, int col, int aRows, int aCols) const {
    MatrixDesc m = MatrixDesc(aRows, aCols);

    assert(row + aRows <= dims.rows && col + aCols <= dims.cols);

    for (int i = 0; i < aRows; i++) {
      for (int j = 0; j < aCols; j++) {
        m.setDataFloat(i, j, getDataFloat(row + i, col + j));
      }
    }

    return m;
  }

  decltype(dims) getDims() { return dims; }

  void transpose_() {
    MatrixDesc tmpTransposed = MatrixDesc(dims.cols, dims.rows);

    for (int i = 0; i < dims.rows; i++) {
      for (int j = 0; j < dims.cols; j++) {
        tmpTransposed.setDataFloat(j, i, getDataFloat(i, j));
      }
    }
    for (int i = 0; i < dims.cols * dims.rows; i++) {
      setDataFloat(i, tmpTransposed.getDataFloat(i));
    }
    dims.cols = tmpTransposed.dims.cols;
    dims.rows = tmpTransposed.dims.rows;
  }

  MatrixDesc transpose() const {
    MatrixDesc result = MatrixDesc(dims.cols, dims.rows);
    for (int i = 0; i < dims.cols; i++) {
      for (int j = 0; j < dims.rows; j++) {
        result.setDataFloat(i, j, getDataFloat(j, i));
      }
    }
    return result;
  }

  void fill_contiguous_values_() {
    for (int i = 0; i < dims.rows * dims.cols; i++) {
      setDataFloat(i, T(i));
    }
  }

  void fill_(T val) {
    for (int i = 0; i < dims.rows * dims.cols; i++) {
      setDataFloat(i, val);
    }
  }

  void multiplyScalar_(T val) {
    for (int i = 0; i < dims.rows * dims.cols; i++) {
      setDataFloat(i, getDataFloat(i) * val);
    }
  }

  MatrixDesc rowMax() const {
    MatrixDesc result = MatrixDesc(dims.rows, 1);
    for (int i = 0; i < dims.rows; i++) {
      T rowMax = -1 * std::numeric_limits<T>::infinity();
      for (int j = 0; j < dims.cols; j++) {
        rowMax = std::max(rowMax, getDataFloat(i, j));
      }
      result.setDataFloat(i, 0, rowMax);
    }
    return result;
  }

  static MatrixDesc max(const MatrixDesc &a, const MatrixDesc &b) {
    MatrixDesc result = MatrixDesc(a.dims.rows, 1);
    for (int i = 0; i < result.dims.rows; i++) {
      result.setDataFloat(i, 0,
                          std::max(a.getDataFloat(i, 0), b.getDataFloat(i, 0)));
    }
    return result;
  }

  void randomInitMatrix_(bool stable = true) {
    for (uint32_t j = 0; j < dims.cols * dims.rows; ++j) {
      if (stable) {
        double randomNumber = ((rand() & 0x3) - 1.0f) / 8.0f;
        setDataFloat(j, (T)randomNumber);
      } else {
        double randomNumber = ((double)rand() / (RAND_MAX));
        setDataFloat(j, (T)randomNumber);
      }
    }
  }

  void pasteSubMatrix(const MatrixDesc &other, int row, int col,
                      int op = MATRIX_OP_OVERWRITE) {
    assert(row + other.dims.rows <= dims.rows &&
           col + other.dims.cols <= dims.cols);

    for (int i = 0; i < other.dims.rows; i++) {
      for (int j = 0; j < other.dims.cols; j++) {
        if (op == MATRIX_OP_OVERWRITE) {
          setDataFloat(row + i, col + j, other.getDataFloat(i, j));
        }

        if (op == MATRIX_OP_ADD) {
          setDataFloat(row + i, col + j,
                       getDataFloat(row + i, col + j) +
                           other.getDataFloat(i, j));
        }
      }
    }
  }

  MatrixDesc diff(const MatrixDesc &other) const {
    MatrixDesc result = MatrixDesc(dims.rows, dims.cols);
    for (int i = 0; i < result.dims.rows; i++) {
      for (int j = 0; j < result.dims.cols; j++) {
        result.setDataFloat(i, j,
                            getDataFloat(i, j) - other.getDataFloat(i, j));
      }
    }
    return result;
  }

  double mse(const MatrixDesc &other) const {
    double acc = 0.0;
    for (int i = 0; i < dims.rows; i++) {
      for (int j = 0; j < dims.cols; j++) {
        double diff = getDataFloat(i, j) - other.getDataFloat(i, j);
        acc += diff * diff;
      }
    }
    return acc / (dims.rows * dims.cols);
  }

  void rowSoftmax(MatrixDesc<T> &result, bool normalize = true) const {
    if (normalize) {
      for (int i = 0; i < dims.rows; i++) {
        T rowMax = -1 * std::numeric_limits<T>::infinity();
        for (int j = 0; j < dims.cols; j++) {
          rowMax = std::max(rowMax, getDataFloat(i, j));
        }
        T denominator = T(0);
        for (int j = 0; j < dims.cols; j++) {
          denominator += std::exp(getDataFloat(i, j) - rowMax);
        }
        for (int j = 0; j < dims.cols; j++) {
          T numerator = std::exp(getDataFloat(i, j) - rowMax);
          result.setDataFloat(i, j, numerator / denominator);
        }
      }
    } else {
      for (int i = 0; i < dims.rows; i++) {
        T denominator = T(0);
        for (int j = 0; j < dims.cols; j++) {
          denominator += std::exp(getDataFloat(i, j));
        }
        for (int j = 0; j < dims.cols; j++) {
          T numerator = std::exp(getDataFloat(i, j));
          result.setDataFloat(i, j, numerator / denominator);
        }
      }
    }
  }

  template <typename ResultType = T> MatrixDesc<ResultType> copy() const {
    MatrixDesc<ResultType> result =
        MatrixDesc<ResultType>(dims.rows, dims.cols);
    copy(result);
    return result;
  }

  template <typename ResultType = T>
  void copy(MatrixDesc<ResultType> &other) const {
    assert(dims.cols == other.dims.cols && dims.rows == other.dims.rows

    );
    for (int i = 0; i < dims.rows; i++) {
      for (int j = 0; j < dims.cols; j++) {
        other.setDataFloat(i, j, ResultType(getDataFloat(i, j)));
      }
    }
  }

  template <typename InputType, typename ResultType>
  static void div(const MatrixDesc<InputType> &a,
                  const MatrixDesc<InputType> &b, MatrixDesc<ResultType> &c) {
    for (int i = 0; i < a.dims.rows; i++) {
      for (int j = 0; j < a.dims.cols; j++) {
        c.setDataFloat(
            i, j,
            ResultType(InputType(a.getDataFloat(i, j) / b.getDataFloat(i, j))));
      }
    }
  }

  template <typename InputType, typename ResultType>
  static void multiply(const MatrixDesc<InputType> &a,
                       const MatrixDesc<InputType> &b,
                       MatrixDesc<ResultType> &c) {
    assert(a.dims.cols == b.dims.rows && a.dims.rows == c.dims.rows &&
           b.dims.cols == c.dims.cols);

    for (int i = 0; i < c.dims.rows; i++) {
      for (int j = 0; j < c.dims.cols; j++) {
        ResultType acc = ResultType(0);
        for (int k = 0; k < a.dims.cols; k++) {
          acc += ResultType(
              InputType(a.getDataFloat(i, k) * b.getDataFloat(k, j)));
        }
        c.setDataFloat(i, j, ResultType(acc));
      }
    }
  }

  template <typename ResultType>
  static void copyInto(const MatrixDesc<T> &a, MatrixDesc<ResultType> &b) {
    assert(a.dims.cols == b.dims.cols && a.dims.rows == b.dims.rows);

    for (int i = 0; i < b.dims.rows; i++) {
      for (int j = 0; j < b.dims.cols; j++) {
        b.setDataFloat(i, j, ResultType(a.getDataFloat(i, j)));
      }
    }
  }

  void setDataFloat(uint32_t i, T value) {
    T *_ptr = getPtr();
    _ptr[i] = value;
  }

  T getDataFloat(uint32_t i) const {
    T *_ptr = getPtr();
    return _ptr[i];
  }

  T getDataFloat(int m, int n) const { return getDataFloat(m * dims.cols + n); }

  void setDataFloat(int m, int n, T val) {
    setDataFloat(m * dims.cols + n, val);
  }

  void copyBufferTo(void *dstPtr) {
    std::memcpy(dstPtr, (void *)getPtr(), dims.cols * dims.rows * sizeof(T));
  }

  void printMatrix(std::ostream &log = std::cout) const {
    std::cout << std::setprecision(8) << std::fixed;
    log << "Matrix: dims=" << dims.rows << "x" << dims.cols
        << ", element size (bytes)=" << sizeof(T) << std::endl;
    for (int j = 0; j < dims.cols; j++) {
      log << "\t";
      if (j > 0) {
        log << "\t";
      }
      log << FMT_BEG "1;32;4m" << j << FMT_CLR;
    }
    log << std::endl;
    for (int i = 0; i < dims.rows; i++) {
      log << FMT_BEG "1;32;4m" << i << FMT_CLR;
      for (int j = 0; j < dims.cols; j++) {
        log << "\t" << getDataFloat(i, j) << " ";
      }
      log << std::endl;
    }
    std::cout << std::defaultfloat;
  }

  void printMatrixFlat(std::ostream &log = std::cout) const {
    log << "Matrix: dims=" << dims.rows << "x" << dims.cols
        << ", element size (bytes)=" << sizeof(T) << std::endl;
    for (int j = 0; j < dims.cols * dims.rows; j++) {
      log << "\t" << getDataFloat(j) << " ";
    }
    log << std::endl;
  }
};

/*
Computes flash attention on CPU.
*/
template <typename InputType, typename IntermediateType,
          typename AccumulatorType>
void flashAttentionHost(const MatrixDesc<InputType> &Q,
                        const MatrixDesc<InputType> &Kt,
                        const MatrixDesc<InputType> &V,
                        MatrixDesc<AccumulatorType> &O,
                        MatrixDesc<AccumulatorType> &L,
                        MatrixDesc<IntermediateType> &S, int Br, int Bc,
                        IntermediateType scaling = IntermediateType(1.0),
                        std::ostream &log = std::cout) {

  log << "Host computation (flash attention)... Types "
         "(Input/Intermediate/Accumulator): "
      << typeid(InputType).name() << " " << typeid(IntermediateType).name()
      << " " << typeid(AccumulatorType).name() << std::endl;

  int N = Q.dims.rows;
  int d = Q.dims.cols;

  assert(Kt.dims.rows == d && Kt.dims.cols == N && V.dims.cols == d &&
         V.dims.rows == N && O.dims.cols == d && O.dims.rows == N &&
         N % Br == 0 && N % Bc == 0);

  int nTilesI = N / Br;
  int nTilesJ = N / Bc;

  for (int tileI = 0; tileI < nTilesI; tileI++) {
    MatrixDesc Qi = Q.extractSubMatrix(tileI * Br, 0, Br, d);

    MatrixDesc<IntermediateType> mi_j = MatrixDesc<IntermediateType>(
        Br, 1, -1 * std::numeric_limits<IntermediateType>::infinity());
    MatrixDesc<IntermediateType> li_j =
        MatrixDesc<IntermediateType>(Br, 1, IntermediateType(0.0));
    MatrixDesc<AccumulatorType> Oi_j =
        MatrixDesc<AccumulatorType>(Br, d, AccumulatorType(0.0));

    for (int tileJ = 0; tileJ < nTilesJ; tileJ++) {
      MatrixDesc<InputType> Kj = Kt.extractSubMatrix(0, tileJ * Bc, d, Bc);
      MatrixDesc<InputType> Vj = V.extractSubMatrix(tileJ * Bc, 0, Bc, d);

      MatrixDesc<IntermediateType> mi_jm1 = mi_j.copy();
      MatrixDesc<IntermediateType> li_jm1 = li_j.copy();
      MatrixDesc<AccumulatorType> Oi_jm1 = Oi_j.copy();

      // Computation

      // Si_j = Qi @ Kj
      MatrixDesc<AccumulatorType> tmpSi_j = MatrixDesc<AccumulatorType>(
          Qi.dims.rows, Kj.dims.cols, AccumulatorType(0.0));
      MatrixDesc<AccumulatorType>::template multiply<InputType,
                                                     AccumulatorType>(Qi, Kj,
                                                                      tmpSi_j);
      MatrixDesc<IntermediateType> Si_j =
          tmpSi_j.template copy<IntermediateType>();
      Si_j.multiplyScalar_(scaling);

      // Save S
      S.pasteSubMatrix(Si_j, Br * tileI, Bc * tileJ);

      // maxSi_j = rowmax(Si_j)
      MatrixDesc<IntermediateType> maxSi_j = Si_j.rowMax();

      // mi_j = max(rowmax(Si_j), mi_jm1)
      mi_j = MatrixDesc<IntermediateType>::max(mi_jm1, maxSi_j);

      // Pijt = exp(Si_j - mi_j)
      MatrixDesc<IntermediateType> Pijt = MatrixDesc<IntermediateType>(Br, Bc);
      for (int i = 0; i < Br; i++) {
        for (int j = 0; j < Bc; j++) {
          Pijt.setDataFloat(i, j,
                            IntermediateType(std::exp(
                                Si_j.getDataFloat(i, j) -
                                (IntermediateType)mi_j.getDataFloat(i, 0))));
        }
      }

      // li_j = exp(mi_jm1 - mi_j) * li_jm1 + rowsum(Pijt)
      for (int i = 0; i < Pijt.dims.rows; i++) {
        IntermediateType rowsumPijt = IntermediateType(0.0);
        for (int j = 0; j < Pijt.dims.cols; j++) {
          rowsumPijt += (IntermediateType)Pijt.getDataFloat(i, j);
        }
        li_j.setDataFloat(i, 0,
                          IntermediateType(std::exp(mi_jm1.getDataFloat(i, 0) -
                                                    mi_j.getDataFloat(i, 0)) *
                                               li_jm1.getDataFloat(i, 0) +
                                           rowsumPijt));
      }

      // O_ij = exp(mi_jm1 - mi_j) * Oi_jm1
      for (int i = 0; i < Oi_j.dims.rows; i++) {
        AccumulatorType scaleFactor = AccumulatorType(
            std::exp(mi_jm1.getDataFloat(i, 0) - mi_j.getDataFloat(i, 0)));
        for (int j = 0; j < Oi_j.dims.cols; j++) {
          Oi_j.setDataFloat(i, j, scaleFactor * Oi_jm1.getDataFloat(i, j));
        }
      }

      // Oi_j = diag(exp(mi_jm1 - mi_j)) @ Oi_jm1 + Pijt @ Vj
      MatrixDesc<InputType> tmp_Pijt = Pijt.template copy<InputType>();
      MatrixDesc<AccumulatorType> PijtVj =
          MatrixDesc<AccumulatorType>(Pijt.dims.rows, Vj.dims.cols);
      MatrixDesc<AccumulatorType>::template multiply<InputType,
                                                     AccumulatorType>(
          tmp_Pijt, Vj, PijtVj);

      for (int i = 0; i < Oi_j.dims.rows; i++) {
        AccumulatorType scaleFactor = AccumulatorType(
            std::exp(mi_jm1.getDataFloat(i, 0) - mi_j.getDataFloat(i, 0)));
        for (int j = 0; j < Oi_j.dims.cols; j++) {
          Oi_j.setDataFloat(
              i, j, Oi_j.getDataFloat(i, j) + PijtVj.getDataFloat(i, j));
        }
      }

    } // End loop tileJ

    // Oi = inv(diag(li_j)) @ Oi_j
    MatrixDesc<AccumulatorType> Oi = MatrixDesc<AccumulatorType>(Br, d);
    for (int i = 0; i < Oi_j.dims.rows; i++) {
      for (int j = 0; j < Oi_j.dims.cols; j++) {
        Oi.setDataFloat(i, j,
                        Oi_j.getDataFloat(i, j) *
                            AccumulatorType(IntermediateType(1.0) /
                                            li_j.getDataFloat(i, 0)));
      }

      // Li = mi_j + log(li_j)
      L.setDataFloat(i, 0,
                     AccumulatorType(mi_j.getDataFloat(i, 0) +
                                     std::log(li_j.getDataFloat(i, 0))));
    }

    O.pasteSubMatrix(Oi, Br * tileI, 0);
  }
}

template <typename InputType, typename IntermediateType,
          typename AccumulatorType>
void attentionHost(const MatrixDesc<InputType> &Q,
                   const MatrixDesc<InputType> &Kt,
                   const MatrixDesc<InputType> &V,
                   MatrixDesc<IntermediateType> &S,
                   MatrixDesc<AccumulatorType> &O,
                   MatrixDesc<IntermediateType> &P, bool causal = false,
                   IntermediateType scaling = IntermediateType(1.0),
                   std::ostream &log = std::cout) {

  log << "Host computation (naive)... Types (Input/Intermediate/Accumulator): "
      << typeid(InputType).name() << " " << typeid(IntermediateType).name()
      << " " << typeid(AccumulatorType).name() << std::endl;

  // Type:quantity

  // Q:Inp, Kt:Inp -> QKtResult:Acc
  MatrixDesc QKtResult =
      MatrixDesc<AccumulatorType>(S.getDims().rows, S.getDims().cols);
  MatrixDesc<AccumulatorType>::template multiply<InputType, AccumulatorType>(
      Q, Kt, QKtResult);
  // QKtResult:Acc -> S:Int
  QKtResult.template copy<IntermediateType>(S);

  // S:Int -> S:Int
  for (int i = 0; i < S.getDims().rows; i++) {
    for (int j = 0; j < S.getDims().cols; j++) {
      S.setDataFloat(i, j, S.getDataFloat(i, j) * scaling);
      if (causal) {
        if (j > i) {
          S.setDataFloat(i, j, IntermediateType(-1.0 / 0.0));
        }
      }
    }
  }

  // S:Int -> P:Int
  S.rowSoftmax(P);

  // P:Int -> _P:Inp
  MatrixDesc<InputType> _P =
      MatrixDesc<InputType>(P.getDims().rows, P.getDims().cols);
  P.template copy<InputType>(_P);

  // _P:Inp, V:Inp -> O:Acc
  MatrixDesc<AccumulatorType>::template multiply<InputType, AccumulatorType>(
      _P, V, O);
}

// Manages a memory buffer on the host and the device (optional).
// Memory resouces should be exclusively owned by the object.
// Declaration provided because VulkanContext also uses DeviceMappedArrays.
class DeviceMappedArray : public Array {
private:
  std::shared_ptr<VulkanContext> context;
  VkDeviceMemory hostMemory;
  VkDeviceMemory deviceMemory;
  VkDeviceSize bufferSize;
  VkBuffer hostBuffer;
  VkBuffer deviceBuffer;
  // Host memory buffer is allocated through vulkan, and the host accessible
  // pointer (ptr) is set by vulkan.
  bool vkHostManaged = false;
  // Device memory buffer is allocated through vulkan.
  bool vkDeviceManaged = false;
  bool hostOnly = false;
  void *const ptr = nullptr;

public:
  const static VkBufferUsageFlags SSBO_flags =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | // Support load, store and atomic ops
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | // Supports load ops
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |   // Data can be copied into the buffer
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |   // Data can be read from the buffer
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; // The buffer's address can be
                                                 // queried and used in a shader

  const static VkBufferUsageFlags UBO_flags =
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  DeviceMappedArray() = delete;
  DeviceMappedArray(const DeviceMappedArray &) = delete;
  DeviceMappedArray(std::shared_ptr<VulkanContext> aContext, uint64_t size,
                    bool aHostOnly = false,
                    VkBufferUsageFlags bufferUsageFlags = SSBO_flags);
  void *const getPtr() const override;
  size_t getSize() const override;
  void copyDataFrom(Array &src) const override;
  VkBuffer getDeviceBuffer();
  VkBuffer getHostBuffer();
  ~DeviceMappedArray();
};

/*
A context that manages vulkan resources.
*/
class VulkanContext {
private:
  VkInstance instance;
  VkDebugUtilsMessengerEXT messenger;
  VkPhysicalDevice physicalDevice;
  VkPhysicalDeviceProperties physicalDeviceProperties;
  VkPhysicalDeviceMemoryProperties memoryProperties;
  VkDevice device;
  int queueFamilyIndex;
  VkQueue queue;
  uint32_t numberOfDescriptorSets = 1;
  uint32_t numberOfDescriptors = 1;
  VkDescriptorSetLayout descriptorSetLayout;
  VkPipelineLayout pipelineLayout;
  VkDescriptorPool descriptorPool;
  VkDescriptorSet descriptorSet;
  VkCommandPool commandPool;
  VkCommandBuffer commandBuffers[3];

  const std::vector<const char *> validationLayers = {
      "VK_LAYER_KHRONOS_validation"};

#ifdef DEBUG_MODE
  const bool enableValidationLayers = true;
#else
  const bool enableValidationLayers = false;
#endif

  void initVulkanEnvironment() {
    createInstance();
    addInstanceCallbacks();
    pickPhysicalDevice();
    queryPhysicalDeviceProperties();
    createLogicalDevice();
    createComputeDescriptorSetLayout();
    createPipelineLayout();
    createDescriptorPool();
    allocateDescriptorSets();
    createCommandPool();
    createCommandBuffers();
    loadShaderFiles("shaders");
  }

  bool checkValidationLayerSupport() const {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers) {
      bool layerFound = false;

      for (const auto &layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }

    return true;
  }

  VkValidationFeaturesEXT getEnabledValidationFeatures() const {
    std::vector<VkValidationFeatureEnableEXT> validation_feature_enables = {
        VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
    VkValidationFeaturesEXT validation_features{
        VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    validation_features.enabledValidationFeatureCount = 1;
    validation_features.pEnabledValidationFeatures =
        validation_feature_enables.data();

    return validation_features;
  }

  static VkBool32
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                VkDebugUtilsMessageTypeFlagsEXT types,
                const VkDebugUtilsMessengerCallbackDataEXT *data,
                void *pUserData) {

    std::cout << "Debug: " << data->pMessage << std::endl;

    return false;
  }

  void addInstanceCallbacks() {
    VkDebugReportCallbackEXT debugCallbackHandle;
    VkDebugUtilsMessengerCreateInfoEXT createInfo;

    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
      CHECK_RESULT(func(instance, &createInfo, nullptr, &messenger));
    } else {
      throw std::runtime_error(
          "Extension vkCreateDebugUtilsMessengerEXT not present!");
    }
  }

  void createInstance() {
    VkApplicationInfo applicationInfo = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        NULL,
        "Cooperative matrix performance test",
        1,
        "none",
        0,
        VK_MAKE_VERSION(1, 3, 0),
    };

    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }

    const char *enabledInstanceExtensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        VK_EXT_LAYER_SETTINGS_EXTENSION_NAME};

    VkInstanceCreateInfo instanceCreateInfo = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        NULL,
        0,
        &applicationInfo,
        0,
        NULL,
        2,
        enabledInstanceExtensions,
    };

    if (enableValidationLayers) {
      instanceCreateInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();

      // TODO: Enable validation features for printf.
      // Currently validation layer triggers an error when enabled along with
      // coopmats.
      /*
      std::vector<VkValidationFeatureEnableEXT>  validation_feature_enables =
      {VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};

      VkValidationFeaturesEXT
      validation_features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
      validation_features.enabledValidationFeatureCount = 1;
      validation_features.pEnabledValidationFeatures =
      validation_feature_enables.data();

      //validationFeatures.pNext = instanceCreateInfo.pNext;
      instanceCreateInfo.pNext = &validation_features;
      */

    } else {
      instanceCreateInfo.enabledLayerCount = 0;
    }

    CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, NULL, &instance));
  }

  void pickPhysicalDevice() {
    uint32_t numPhysicalDevices = 0;
    vector<VkPhysicalDevice> physicalDevices;

    CHECK_RESULT(
        vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, NULL));

    physicalDevices.resize(numPhysicalDevices);
    CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &numPhysicalDevices,
                                            &physicalDevices[0]));

    int physicalDeviceIndex = -1;

    std::cout << "Available devices: " << numPhysicalDevices << std::endl;
    for (uint32_t i = 0; i < numPhysicalDevices; ++i) {
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(physicalDevices[i], &deviceProperties);
      std::cout << "* device id=" << i << ": " << deviceProperties.deviceName
                << std::endl;
    }
    for (uint32_t i = 0; i < numPhysicalDevices; ++i) {

      uint32_t numExtensions = 0;
      vector<VkExtensionProperties> extensions;

      CHECK_RESULT(vkEnumerateDeviceExtensionProperties(
          physicalDevices[i], NULL, &numExtensions, NULL));

      extensions.resize(numExtensions);
      CHECK_RESULT(vkEnumerateDeviceExtensionProperties(
          physicalDevices[i], NULL, &numExtensions, &extensions[0]));

      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(physicalDevices[i], &deviceProperties);

      for (uint32_t j = 0; j < numExtensions; ++j) {
        const char *selDevName = std::getenv("DEVICE_NAME");
        if (selDevName != nullptr) {
          if (strstr(deviceProperties.deviceName, selDevName) == NULL) {
            continue;
          }
        }
        if (strcmp(extensions[j].extensionName,
                   VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
          std::cout << "Selecting device " << i << std::endl;
          physicalDeviceIndex = i;
          break;
        }
      }
      if (physicalDeviceIndex != -1) {
        break;
      }
    }

    if (physicalDeviceIndex == -1) {
      throw std::invalid_argument(
          "Could not find a device that supports coperative matrix extension!");
    }
    physicalDevice = physicalDevices[physicalDeviceIndex];

    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
  }

  vector<VkCooperativeMatrixPropertiesKHR>
  queryCooperativeMatrixProperties() const {
    // Query the list of supported cooperative matrix multiply sizes/types.
    uint32_t numCooperativeMatrixProperties = 0;
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
        pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
            (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
                vkGetInstanceProcAddr(
                    instance,
                    "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

    CHECK_RESULT(pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
        physicalDevice, &numCooperativeMatrixProperties, NULL));

    vector<VkCooperativeMatrixPropertiesKHR> cooperativeMatrixProperties;

    cooperativeMatrixProperties.resize(numCooperativeMatrixProperties);
    for (uint32_t i = 0; i < numCooperativeMatrixProperties; ++i) {
      cooperativeMatrixProperties[i].sType =
          VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
      cooperativeMatrixProperties[i].pNext = NULL;
    }

    CHECK_RESULT(pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
        physicalDevice, &numCooperativeMatrixProperties,
        &cooperativeMatrixProperties[0]));

    return cooperativeMatrixProperties;
  }

  void queryPhysicalDeviceProperties() {
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

    subgroupProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroupProperties.pNext = NULL;

    VkPhysicalDeviceProperties2 physicalDeviceProperties;
    physicalDeviceProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physicalDeviceProperties.pNext = &subgroupProperties;

    vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties);
  }

  void createLogicalDevice() {
    uint32_t numQueueFamilies = 0;
    vector<VkQueueFamilyProperties> queueFamilies;

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies,
                                             NULL);

    queueFamilies.resize(numQueueFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies,
                                             &queueFamilies[0]);

    // int queueFamilyIndex = -1;

    for (uint32_t i = 0; i < numQueueFamilies; ++i) {
      if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        queueFamilyIndex = i;
        break;
      }
    }
    if (queueFamilyIndex == -1) {
      throw std::invalid_argument("Could not find compute queue!");
      return;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        NULL,
        0,
        (uint32_t)queueFamilyIndex,
        1,
        &queuePriority,
    };

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, NULL,
        VK_TRUE,  // cooperativeMatrix
        VK_FALSE, // cooperativeMatrixRobustBufferAccess
    };

    VkPhysicalDeviceVulkan11Features vulkan11Features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        &coopMatFeatures};
    vulkan11Features.storageBuffer16BitAccess = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vulkan12Features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        &vulkan11Features};
    vulkan12Features.bufferDeviceAddress = VK_TRUE;
    vulkan12Features.shaderFloat16 = VK_TRUE;
    vulkan12Features.shaderInt8 = VK_TRUE;
    vulkan12Features.vulkanMemoryModel = VK_TRUE;
    vulkan12Features.vulkanMemoryModelDeviceScope = VK_TRUE;

    VkPhysicalDeviceVulkan13Features vulkan13Features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        &vulkan12Features};
    vulkan13Features.subgroupSizeControl = VK_TRUE;
    vulkan13Features.computeFullSubgroups = VK_TRUE;

    const char *enabledDeviceExtensions[] = {
        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
    };
    VkDeviceCreateInfo deviceCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        &vulkan13Features,
        0,
        1,
        &deviceQueueCreateInfo,
        0,
        NULL,
        sizeof(enabledDeviceExtensions) / sizeof(enabledDeviceExtensions[0]),
        enabledDeviceExtensions,
        NULL,
    };

    CHECK_RESULT(
        vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device));

    vkGetDeviceQueue(device, (uint32_t)queueFamilyIndex, 0, &queue);
  }

  void createComputeDescriptorSetLayout() {
    /*
    Create descriptor set layout with one binding containing a descriptor set of
    a single element.
    */
    VkDescriptorSetLayoutBinding layoutBindings[1] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         numberOfDescriptors, // Number of descriptors
         VK_SHADER_STAGE_COMPUTE_BIT, NULL}};

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        NULL,
        0, // Flags
        ARRAY_LENGTH(layoutBindings),
        layoutBindings,
    };

    CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
  }

  void createPipelineLayout() {
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        NULL,
        0,                    // Flags
        1,                    // Number of descriptor sets
        &descriptorSetLayout, // The array of descriptor sets
        0,                    // Number of constants
        NULL                  // The constants
    };

    CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL,
                                        &pipelineLayout));
  }

  void createDescriptorPool() {
    // We have a single uniform buffer in our descriptor set layout that stores
    // the device addresses of the matrices needeed by the algorithm.
    VkDescriptorPoolSize poolSizes[1] = {{
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // type
        1                                  // number of descriptors
    }};

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        NULL,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        1, // Maximum number of descriptor sets
        ARRAY_LENGTH(poolSizes),
        poolSizes,
    };

    CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL,
                                        &descriptorPool));
  }

  void allocateDescriptorSets() {
    VkDescriptorSetAllocateInfo setAllocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        NULL,
        descriptorPool,
        numberOfDescriptorSets,
        &descriptorSetLayout,
    };

    CHECK_RESULT(
        vkAllocateDescriptorSets(device, &setAllocateInfo, &descriptorSet));
  }

  void createCommandPool() {
    VkCommandPoolCreateInfo commandPoolCreateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        NULL,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        (uint32_t)queueFamilyIndex,
    };

    CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL,
                                     &commandPool));
  }

  void createCommandBuffers() {
    // The command buffers, one for initializing buffers, one for compute, one
    // for reading back the results. This lets us time the compute work more
    // precisely.
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        NULL,
        commandPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        3,
    };

    CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo,
                                          commandBuffers));
  }

  void loadShaderFiles(std::string path) {
    std::cout << "Loading shaders from: " << path << std::endl;
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
      if (entry.path().extension() == ".spv") {
        std::cout << "* loading shader: " << entry.path() << std::endl;
        vector<char> spirv = VulkanHelper::loadSPIRV(entry.path().c_str());
        shaderFiles[entry.path()] = spirv;
      }
    }
  }

  void cleanup() {
    vkDeviceWaitIdle(device);
    vkDestroyCommandPool(device, commandPool, NULL);

    vkDestroyDescriptorPool(device, descriptorPool, NULL);
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);

    vkDestroyDevice(device, NULL);

    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
      func(instance, messenger, nullptr);
    }
    vkDestroyInstance(instance, NULL);
  }

public:
  uint32_t numOfParallelSubgroups = 4;
  VkPhysicalDeviceSubgroupProperties subgroupProperties;
  std::map<std::string, std::vector<char>> shaderFiles;

  VkDevice getDevice() { return device; }

  VkPhysicalDeviceProperties getPhysicalDeviceProperties() {
    return physicalDeviceProperties;
  }

  VkPhysicalDeviceMemoryProperties getMemoryProperties() {
    return memoryProperties;
  }

  VulkanContext() { initVulkanEnvironment(); }

  ~VulkanContext() { cleanup(); }

  VkCommandBuffer *getCommandBuffers() { return commandBuffers; }

  VkCommandBuffer getComputeCommandBuffer() { return commandBuffers[1]; }

  VkCommandBuffer getHostToDeviceTransferCommandBuffer() {
    return commandBuffers[0];
  }

  VkCommandBuffer getDeviceToHostTransferCommandBuffer() {
    return commandBuffers[2];
  }

  VkQueue getQueue() { return queue; }

  VkPipelineLayout getPipelineLayout() { return pipelineLayout; }

  void bindComputeDescriptorSets(VkCommandBuffer commandBuffer) {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0u, 1, &descriptorSet, 0u, NULL);
  }

  VkCooperativeMatrixPropertiesKHR checkCooperativeMatrixSupport(
      uint32_t MSize, uint32_t NSize, uint32_t KSize, VkComponentTypeKHR AType,
      VkComponentTypeKHR CType, std::ostream &log = std::cout) const {

    // Query supported cooperative matrix parameters
    log << "Supported cooperative matrix types: " << std::endl;
    uint32_t sel = -1;
    vector<VkCooperativeMatrixPropertiesKHR> cooperativeMatrixProperties =
        queryCooperativeMatrixProperties();

    for (uint32_t i = 0; i < cooperativeMatrixProperties.size(); ++i) {
      VkCooperativeMatrixPropertiesKHR *cooperativeMatrixProps =
          &cooperativeMatrixProperties[i];
      if (sel == -1 && cooperativeMatrixProps->AType == AType &&
          cooperativeMatrixProps->CType == CType &&
          cooperativeMatrixProps->MSize == MSize &&
          cooperativeMatrixProps->NSize == NSize &&
          cooperativeMatrixProps->KSize == KSize) {
        sel = i;
        log << FMT_BEG "1;31;4m"
            << "[SELECTED] ";
      }
      log << "Size (MNK): " << cooperativeMatrixProps->MSize << "x"
          << cooperativeMatrixProps->NSize << "x"
          << cooperativeMatrixProps->KSize << " "
          << "Types: (AxB+C=D) \t"
          << VulkanHelper::getTypeName(cooperativeMatrixProps->AType)
          << "\t x \t"
          << VulkanHelper::getTypeName(cooperativeMatrixProps->BType)
          << "\t + \t"
          << VulkanHelper::getTypeName(cooperativeMatrixProps->CType)
          << "\t = \t"
          << VulkanHelper::getTypeName(cooperativeMatrixProps->CType)
          << std::endl;
      if (sel == i) {
        log << FMT_CLR;
      }
    }

    if (sel < 0) {
      std::cerr << "No compatible cooperative matrix op supported."
                << std::endl;
    }

    return cooperativeMatrixProperties[sel];
  }

  void writeMatrixAddressesToDescriptorSet(
      std::vector<std::shared_ptr<DeviceMappedArray>> arrays,
      std::shared_ptr<DeviceMappedArray> paramArray) const {
    PFN_vkGetBufferDeviceAddress pfn_vkGetBufferDeviceAddress =
        (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(
            device, "vkGetBufferDeviceAddress");

    for (int i = 0; i < arrays.size(); ++i) {
      if (arrays[i] != nullptr) {
        VkBufferDeviceAddressInfo info = {
            VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            NULL,
            arrays[i]->getDeviceBuffer(),
        };
        VkDeviceAddress addr = pfn_vkGetBufferDeviceAddress(device, &info);
        ((VkDeviceAddress *)paramArray->getPtr())[i] = addr;
      }
    }
  }

  void updateDescriptors(std::shared_ptr<DeviceMappedArray> paramArray) {
    VkDescriptorBufferInfo bufferDescriptor = {paramArray->getHostBuffer(), 0,
                                               paramArray->getSize()};

    // Write the buffer to the descriptor set.
    // We write it into the first element of the first binding
    VkWriteDescriptorSet writeDescriptorset = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        NULL,
        descriptorSet,
        0, // Select the first bindig
        0, // Select the first descriptor in the first binding
        1, // We write to only descriptor
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        NULL,
        &bufferDescriptor, // The array of descriptors we write (single element)
        NULL,
    };

    vkUpdateDescriptorSets(device, 1, &writeDescriptorset, 0,
                           NULL); // Write 1 desciptor and copy 0
  }
};

DeviceMappedArray::DeviceMappedArray(std::shared_ptr<VulkanContext> aContext,
                                     uint64_t size, bool aHostOnly,
                                     VkBufferUsageFlags bufferUsageFlags) {
  context = aContext;
  bufferSize = size;
  hostOnly = aHostOnly;

  uint32_t hostMemorySettings = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

  if (!hostOnly) {
    // Allocate and assign a device buffer
    uint32_t deviceMemorySettings = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VulkanHelper::createArray(
        context->getDevice(), context->getMemoryProperties(), bufferSize,
        deviceBuffer, deviceMemory, deviceMemorySettings, bufferUsageFlags);
    vkDeviceManaged = true;
  }

  // Allocate and assign a host buffer
  VulkanHelper::createArray(
      context->getDevice(), context->getMemoryProperties(), bufferSize,
      hostBuffer, hostMemory, hostMemorySettings, bufferUsageFlags);

  CHECK_RESULT(vkMapMemory(context->getDevice(), hostMemory, 0, bufferSize, 0,
                           const_cast<void **>(&ptr)));

  vkHostManaged = true;
}

void *const DeviceMappedArray::getPtr() const { return ptr; }

size_t DeviceMappedArray::getSize() const { return bufferSize; }

void DeviceMappedArray::copyDataFrom(Array &src) const {
  std::memcpy(ptr, src.getPtr(), std::min(bufferSize, src.getSize()));
}

VkBuffer DeviceMappedArray::getDeviceBuffer() {
  if (hostOnly) {
    throw std::invalid_argument(
        "Requested a device buffer of a host only array.");
  }
  return deviceBuffer;
}

VkBuffer DeviceMappedArray::getHostBuffer() { return hostBuffer; }

DeviceMappedArray::~DeviceMappedArray() {
  vkUnmapMemory(context->getDevice(), hostMemory);
  if (vkHostManaged) {
    vkDestroyBuffer(context->getDevice(), hostBuffer, NULL);
    vkFreeMemory(context->getDevice(), hostMemory, NULL);
  }
  if (vkDeviceManaged) {
    vkDestroyBuffer(context->getDevice(), deviceBuffer, NULL);
    vkFreeMemory(context->getDevice(), deviceMemory, NULL);
  }
}

struct KernelParamsForward {
  uint32_t inputSize;
  uint32_t strideB, strideH, strideS, strideD;
  uint32_t lM, lN, lK;
  uint32_t Vr, Vc;
  uint32_t Ur, Uc;
  uint32_t causal;
  float scaling;
  uint32_t subgroupSize, localSize;
};

/* Parameters for measuring the performance. */
struct PerfParams {
  uint32_t nReps;
  uint32_t warmUpCycles;
  uint64_t elapsedMicrosecs; /* Stores the result. */
};

/*
Creates a vulkan compute pipeline. The computation can be invoked multiple times
with different arrays but having the same parameters (e.g. strides, input size,
or component types).

It uses a previously created Vulkan context that provides the resouces (e.g.
physical device, logical device, buffer descritprs, compute queues, commands).
*/
class FusedSDPA {
private:
  VkShaderModule shaderModule;
  VkPipeline pipeline;

public:
  std::shared_ptr<VulkanContext> context;
  uint32_t elemSizeA;
  uint32_t elemSizeS;
  uint32_t elemSizeC;
  uint32_t Br;
  uint32_t Bc;
  uint32_t b;
  uint32_t h;
  uint32_t s;
  uint32_t d;
  uint32_t N;
  uint32_t numberOfWorkgroups;
  uint32_t totalElements;
  bool verbose = false;

  FusedSDPA(std::shared_ptr<VulkanContext> aContext, uint32_t aTotalElements,
            std::vector<uint32_t> strides, uint32_t aElemSizeA,
            uint32_t aElemSizeS, uint32_t aElemSizeC, float scaling,
            bool causal, std::ostream &log = std::cout)
      : context(aContext), elemSizeA(aElemSizeA), elemSizeS(aElemSizeS),
        elemSizeC(aElemSizeC), totalElements(aTotalElements) {

    std::vector<uint32_t> dims = getDims(strides, totalElements);
    b = dims[0];
    h = dims[1];
    s = dims[2];
    d = dims[3];
    uint32_t inputSize = b * h * s * d;
    N = b * h * s;

    uint32_t Vr, Vc, Ur, Uc, Wr, Wc;

    uint32_t MSize = 16;
    uint32_t NSize = 16;
    uint32_t KSize = 16;

    // Warptile size in cooperative matrices
    uint32_t Vd = d / KSize;

    // Blocktile size in warptiles
    Br = Ur * Wr;
    Bc = Uc * Wc;

    optimizeParameters(MSize, NSize, Vr, Vc, Ur, Uc, s, d, Wr, Wc, Br, Bc);

    // Number of warptiles (unused)
    uint32_t Tr = s / Br;
    uint32_t Tc = s / Bc;

    // Total number of blocks in the sequence length dimension
    numberOfWorkgroups = (b * h * s) / Br;

    log << "Elem sizes: (input, intermediate, accumulator): (" << elemSizeA
        << ", " << elemSizeS << ", " << elemSizeC << ")" << std::endl;
    log << "Full problem size:" << std::endl;
    log << "* Dims: (b, h, s, d):     (" << b << ", " << h << ", " << s << ", "
        << d << ")" << std::endl;
    log << "* Strides: (b, h, s, d):  (" << strides[0] << ", " << strides[1]
        << ", " << strides[2] << ", " << strides[3] << ")" << std::endl;

    log << "Tiling parameters (warp, block, coopmats): " << std::endl;
    log << "* Number of coopmats per warptile (Vr, Vc, Vd):   (" << Vr << ", "
        << Vc << ", " << Vd << ")" << std::endl;
    log << "* Number of warptiles per block (Ur, Uc):         (" << Ur << ", "
        << Uc << ")" << std::endl;
    log << "* Number of blocks per input (Tr, Tc):            (" << Tr << ", "
        << Tc << ")" << std::endl;

    log << "Tile sizes (coopmat, warp, block, input):" << std::endl;
    log << "* Coopmat tile size: (Cr, Cc, Cd):    (" << MSize << ", " << NSize
        << ", " << KSize << ")" << std::endl;
    log << "* Warptile edge size: (Wr, Wc):       (" << Wr << ", " << Wc << ")"
        << std::endl;
    log << "* Blocktile edge size: (Br, Bc):      (" << Br << ", " << Bc << ")"
        << std::endl;
    log << "* Tile size: (s,d):                   (" << s << ", " << d << ")"
        << std::endl;

    log << "Number of workgroups: " << numberOfWorkgroups << std::endl;

    if (s < Br || s < Bc || s % Br != 0 || s % Bc != 0) {
      throw std::invalid_argument(
          "Input parameters are not integer multiplies of the blocktile "
          "parameters (N is not compatible with either Bc or Br).");
    }

    if (d < KSize || d % KSize != 0) {
      throw std::invalid_argument(
          "Input parameters are not integer multiplies of the coopmat size (d "
          "is not compatible with Cd).");
    }

    if (NSize != KSize) {
      throw std::invalid_argument(
          "Selected coopmat problem size does not allow instant double "
          "multiplication (Cd not equal Cc)");
    }

    if (Wr < MSize || Wc < NSize || Wr % MSize != 0 || Wc % NSize != 0) {
      throw std::invalid_argument(
          "Warptile parameters are not integer multiplies of the coopmat size "
          "(Cr or Cc does not compatible with Wr or Wc)");
    }

    if (Uc != 1) {
      throw std::invalid_argument("Number of warptiles should be 1 until "
                                  "atomics are not introduced in the code.");
    }

    log << "Assumed number of parallel subgroups: "
        << context->numOfParallelSubgroups << std::endl;
    log << "Device defualt subgroup size: "
        << context->subgroupProperties.subgroupSize << std::endl;

    uint32_t localSize = computeLocalSize(Ur, Uc);
    log << "Computed workgroup size: " << localSize << std::endl;

    KernelParamsForward params = initKernelParameters(
        inputSize, strides[0], strides[1], strides[2], strides[3], MSize, NSize,
        KSize, Vr, Vc, Ur, Uc, causal, scaling,
        context->subgroupProperties.subgroupSize, localSize);

    log << "Instantiating forward computation pipeline..." << std::endl;

    /*
    // If we know the type...
    std::string forwardShaderFileName =
        std::string("shaders/sdpa_flash") +
        "-s" + VulkanHelper::getShortTypeName(SType) +
        "-c" + VulkanHelper::getShortTypeName(CType) + ".spv";
    */

    std::string forwardShaderFileName = std::string("shaders/sdpa_flash") +
                                        "-sfp" + std::to_string(elemSizeS * 8) +
                                        "-cfp" + std::to_string(elemSizeC * 8) +
                                        ".spv";
    log << "Using shader: " << forwardShaderFileName << std::endl;

    if (context->shaderFiles.count(forwardShaderFileName) < 1) {
      throw std::invalid_argument("Requested shader is not loaded.");
    }

    shaderModule = createShader(context->shaderFiles[forwardShaderFileName]);
    pipeline = createComputePipelineForward(params, log);
  }

  void optimizeParameters(uint32_t MSize, uint32_t NSize, uint32_t &Vr,
                          uint32_t &Vc, uint32_t &Ur, uint32_t &Uc, uint32_t s,
                          uint32_t d, uint32_t &Wr, uint32_t &Wc, uint32_t &Br,
                          uint32_t &Bc) const {

    uint32_t shmemSize = context->getPhysicalDeviceProperties()
                             .limits.maxComputeSharedMemorySize;
    uint32_t shmemReq = 0;
    uint32_t score = 0;

    for (int i = 1; i < 8; i *= 2) {
      for (int j = 1; j < 8; j *= 2) {
        uint32_t tmpVr = i;
        uint32_t tmpVc = 1;
        uint32_t tmpUr = j;
        uint32_t tmpUc = 1;

        uint32_t tmpBr = tmpUr * tmpVr * MSize;
        uint32_t tmpBc = tmpUc * tmpVc * NSize;

        uint32_t tmpShmemReq = computeSharedMemoryNeeded(
            tmpBr, tmpBc, d, elemSizeA, elemSizeS, elemSizeC);

        uint32_t tmpScore = tmpVr * tmpUr;
        if ((tmpScore > score && tmpShmemReq < shmemSize && tmpBr <= s &&
             tmpBc <= s) ||
            (i == 1 && j == 1)) {
          shmemReq = tmpShmemReq;
          score = tmpScore;
          Vr = tmpVr;
          Vc = tmpVc;
          Ur = tmpUr;
          Uc = tmpUc;
        }
      }
    }

    // Warptile sizes in coopmat tiles
    Wr = Vr * MSize;
    Wc = Vc * NSize;

    // Blocktile size in warptiles
    Br = Ur * Wr;
    Bc = Uc * Wc;

    if (shmemSize < shmemReq) {
      throw std::invalid_argument(
          "Not enough shared memory to execute the kernel (needed: " +
          std::to_string(shmemReq) +
          " vs. available: " + std::to_string(shmemSize) + ")");
    }
  }

  uint32_t computeSharedMemoryNeeded(uint32_t Br, uint32_t Bc, uint32_t d,
                                     uint32_t elemSizeA, uint32_t elemSizeS,
                                     uint32_t elemSizeC) const {
    uint32_t bytesPerUvec4 = 16;
    uint32_t granularityA = bytesPerUvec4 / elemSizeA;
    uint32_t granularityS = bytesPerUvec4 / elemSizeS;
    uint32_t granularityC = bytesPerUvec4 / elemSizeC;

    std::vector<uint32_t> shmemReq;
    shmemReq.push_back((Br * (d + granularityA) / granularityA) *
                       bytesPerUvec4); // Q
    shmemReq.push_back((d * (Bc + granularityA) / granularityA) *
                       bytesPerUvec4); // Kt
    shmemReq.push_back((Bc * (d + granularityA) / granularityA) *
                       bytesPerUvec4); // V
    shmemReq.push_back((Br * (d + granularityC) / granularityC) *
                       bytesPerUvec4); // O
    shmemReq.push_back((Br * (Bc + granularityS) / granularityS) *
                       bytesPerUvec4); // S
    return shmemReq[0] + shmemReq[1] + shmemReq[2] + shmemReq[3] + shmemReq[4];
  }

  uint32_t computeLocalSize(uint32_t Ur, uint32_t Uc) const {
    return std::min(Ur * Uc, context->numOfParallelSubgroups) *
           context->subgroupProperties.subgroupSize;
  }

  KernelParamsForward
  initKernelParameters(uint32_t inputSize, uint32_t strideB, uint32_t strideH,
                       uint32_t strideS, uint32_t strideD, uint32_t MSize,
                       uint32_t NSize, uint32_t KSize, uint32_t Vr, uint32_t Vc,
                       uint32_t Ur, uint32_t Uc, bool causal, float scaling,
                       uint32_t subgroupSize, uint32_t localSize) const {
    KernelParamsForward fwdParams;
    fwdParams.inputSize = inputSize;
    fwdParams.strideB = strideB;
    fwdParams.strideH = strideH;
    fwdParams.strideS = strideS;
    fwdParams.strideD = strideD;
    fwdParams.lM = MSize;
    fwdParams.lN = NSize;
    fwdParams.lK = KSize;
    fwdParams.Vr = Vr;
    fwdParams.Vc = Vc;
    fwdParams.Ur = Ur;
    fwdParams.Uc = Uc;
    fwdParams.causal = causal ? 1 : 0;
    fwdParams.scaling = scaling;
    fwdParams.subgroupSize = subgroupSize;
    fwdParams.localSize = localSize;

    return fwdParams;
  }

  void runComputeShader(VkPipeline pipeline, uint32_t repeatCount,
                        uint32_t groupX, uint32_t groupY,
                        VkCommandBuffer commandBuffer) const {
    VkCommandBufferBeginInfo commandBufferBeginInfo =
        VulkanHelper::getCommandBufferBeginInfo();
    CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    context->bindComputeDescriptorSets(commandBuffer);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    for (uint32_t i = 0; i < repeatCount; ++i) {
      vkCmdDispatch(commandBuffer, groupX, groupY, 1);
    }

    CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
  }

  void executeCommands(VkCommandBuffer commandBuffer,
                       uint64_t *elapsedMicrosecs = nullptr) const {
    auto beginTime = std::chrono::high_resolution_clock::now();
    VkQueue queue = context->getQueue();

    std::vector<VkCommandBuffer> buffers = {commandBuffer};
    VkSubmitInfo submitInfo = VulkanHelper::getSingleCommandSubmitInfo(buffers);
    CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    CHECK_RESULT(vkQueueWaitIdle(queue));

    auto endTime = std::chrono::high_resolution_clock::now();
    uint64_t elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             endTime - beginTime)
                             .count();

    if (elapsedMicrosecs != nullptr) {
      *elapsedMicrosecs = elapsedUs;
    }
  }

  void copyArrays(std::vector<std::shared_ptr<DeviceMappedArray>> arrays,
                  VkCommandBuffer commandBuffer,
                  int direction = COPY_HOST_TO_DEVICE) const {
    assert(direction == COPY_HOST_TO_DEVICE ||
           direction == COPY_DEVICE_TO_HOST);

    VkCommandBufferBeginInfo commandBufferBeginInfo =
        VulkanHelper::getCommandBufferBeginInfo();
    CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    for (uint32_t i = 0; i < arrays.size(); ++i) {
      if (arrays[i] != nullptr) {
        std::shared_ptr<DeviceMappedArray> m = arrays[i];
        VkBufferCopy copy = {0, 0, m->getSize()};
        vkCmdCopyBuffer(commandBuffer,
                        direction == COPY_HOST_TO_DEVICE ? m->getHostBuffer()
                                                         : m->getDeviceBuffer(),
                        direction == COPY_HOST_TO_DEVICE ? m->getDeviceBuffer()
                                                         : m->getHostBuffer(),
                        1, &copy);
      }
    }

    CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
  }

  VkShaderModule createShader(vector<char> spirv) {
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        NULL,
        0,
        spirv.size(),
        (const uint32_t *)&spirv[0],
    };

    CHECK_RESULT(vkCreateShaderModule(
        context->getDevice(), &shaderModuleCreateInfo, NULL, &shaderModule));

    return shaderModule;
  }

  VkPipeline createComputePipelineForward(KernelParamsForward params,
                                          std::ostream &log) const {
    // Specialize the shader with the matrix sizes, strides, and constants.

    log << "Kernel parameters: " << std::endl;
    log << "* inputSize: " << params.inputSize << std::endl;
    log << "* strideB, strideH, strideS, strideD: " << params.strideB << ", "
        << params.strideH << ", " << params.strideS << ", " << params.strideD
        << std::endl;
    log << "* lM, lN, lK: " << params.lM << ", " << params.lN << ", "
        << params.lK << std::endl;
    log << "* Vr, Vc, Ur, Uc: " << params.Vr << ", " << params.Vc << ", "
        << params.Ur << ", " << params.Uc << std::endl;
    log << "* causal, scaling: " << params.causal << ", " << params.scaling
        << std::endl;
    log << "* subgroupSize, localSize: " << params.subgroupSize << ", "
        << params.localSize << std::endl;

    // This should be in order!
    std::vector<uint32_t> entrySizes = {
        sizeof(params.inputSize),    sizeof(params.strideB),
        sizeof(params.strideH),      sizeof(params.strideS),
        sizeof(params.strideD),      sizeof(params.lM),
        sizeof(params.lN),           sizeof(params.lK),
        sizeof(params.Vr),           sizeof(params.Vc),
        sizeof(params.Ur),           sizeof(params.Uc),
        sizeof(params.causal),       sizeof(params.scaling),
        sizeof(params.subgroupSize), sizeof(params.localSize)};

    VkSpecializationMapEntry entries[entrySizes.size()];

    std::vector<uint8_t> specData(sizeof(KernelParamsForward));

    std::memcpy(specData.data(), &params, sizeof(KernelParamsForward));

    uint32_t prefixSum = 0;
    for (int i = 0; i < entrySizes.size(); i++) {
      entries[i].constantID = i;
      entries[i].offset = prefixSum;
      entries[i].size = entrySizes[i];
      prefixSum += entrySizes[i];
    }

    VkSpecializationInfo specInfo = {
        uint32_t(entrySizes.size()),
        entries,
        sizeof(KernelParamsForward),
        specData.data(),
    };

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        NULL,
        params.subgroupSize,
    };

    VkPipelineShaderStageCreateInfo shaderCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        &subgroupSizeCreateInfo,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shaderModule,
        "main",
        &specInfo,
    };

    VkComputePipelineCreateInfo pipelineCreateInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        NULL,
        0,
        shaderCreateInfo,
        context->getPipelineLayout(),
        VK_NULL_HANDLE,
        0};

    VkPipeline pipeline;
    CHECK_RESULT(vkCreateComputePipelines(context->getDevice(), VK_NULL_HANDLE,
                                          1, &pipelineCreateInfo, NULL,
                                          &pipeline));

    return pipeline;
  }

  std::vector<uint32_t> getDims(std::vector<uint32_t> strides,
                                uint32_t totalElements) {
    assert(strides.size() == 4);
    std::vector<uint32_t> dims;
    dims.push_back(totalElements / strides[0]);
    dims.push_back(totalElements / (dims[0] * strides[1]));
    dims.push_back(totalElements / (dims[0] * dims[1] * strides[2]));
    dims.push_back(totalElements / (dims[0] * dims[1] * dims[2] * strides[3]));
    return dims;
  }

  void call(void *buffQ, void *buffKt, void *buffV, void *buffO,
            void *buffS = nullptr, PerfParams *perfParams = nullptr,
            std::ostream &log = std::cout) {

    log << "Device computation..." << std::endl << std::flush;

    std::shared_ptr<HostArray> arrQ =
        std::make_shared<HostArray>(buffQ, totalElements * elemSizeA);
    std::shared_ptr<HostArray> arrKt =
        std::make_shared<HostArray>(buffKt, totalElements * elemSizeA);
    std::shared_ptr<HostArray> arrV =
        std::make_shared<HostArray>(buffV, totalElements * elemSizeA);
    std::shared_ptr<HostArray> arrO =
        std::make_shared<HostArray>(buffO, totalElements * elemSizeC);

    std::shared_ptr<DeviceMappedArray> Q =
        std::make_shared<DeviceMappedArray>(context, elemSizeA * N * d);
    std::shared_ptr<DeviceMappedArray> K =
        std::make_shared<DeviceMappedArray>(context, elemSizeA * N * d);
    std::shared_ptr<DeviceMappedArray> V =
        std::make_shared<DeviceMappedArray>(context, elemSizeA * N * d);
    std::shared_ptr<DeviceMappedArray> O =
        std::make_shared<DeviceMappedArray>(context, elemSizeC * N * d);
    std::shared_ptr<DeviceMappedArray> L =
        std::make_shared<DeviceMappedArray>(context, elemSizeC * N);
#ifdef DEBUG_MODE
    std::shared_ptr<DeviceMappedArray> S =
        std::make_shared<DeviceMappedArray>(context, elemSizeS * N * s);
    std::shared_ptr<DeviceMappedArray> P =
        std::make_shared<DeviceMappedArray>(context, elemSizeS * N * s);
#else
    std::shared_ptr<DeviceMappedArray> S = nullptr;
    std::shared_ptr<DeviceMappedArray> P = nullptr;
#endif

    Q->copyDataFrom(*arrQ);
    K->copyDataFrom(*arrKt);
    V->copyDataFrom(*arrV);

#ifdef DEBUG_MODE
    std::shared_ptr<DeviceMappedArray> QT =
        std::make_shared<DeviceMappedArray>(context, elemSizeA * Br * d);
    std::shared_ptr<DeviceMappedArray> KtT =
        std::make_shared<DeviceMappedArray>(context, elemSizeA * Bc * d);
    std::shared_ptr<DeviceMappedArray> ST =
        std::make_shared<DeviceMappedArray>(context, elemSizeS * Br * Bc);
#else
    std::shared_ptr<DeviceMappedArray> QT = nullptr;
    std::shared_ptr<DeviceMappedArray> KtT = nullptr;
    std::shared_ptr<DeviceMappedArray> ST = nullptr;
#endif

    std::vector<std::shared_ptr<DeviceMappedArray>> kernelParameters = {
        Q, K, V, O, L, S, QT, KtT, ST};

    std::shared_ptr<DeviceMappedArray> paramArray =
        std::make_shared<DeviceMappedArray>(
            context, kernelParameters.size() * sizeof(VkDeviceAddress), true,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    context->writeMatrixAddressesToDescriptorSet(kernelParameters, paramArray);
    context->updateDescriptors(paramArray);

    std::vector<std::shared_ptr<DeviceMappedArray>> inputArrays = {Q, K, V};

    copyArrays(inputArrays, context->getHostToDeviceTransferCommandBuffer(),
               COPY_HOST_TO_DEVICE);
    executeCommands(context->getHostToDeviceTransferCommandBuffer());

    if (perfParams != nullptr) {
      runComputeShader(pipeline, perfParams->nReps, numberOfWorkgroups, 1,
                       context->getComputeCommandBuffer());

      for (int k = 0; k < perfParams->warmUpCycles; k++) {
        executeCommands(context->getComputeCommandBuffer());
      }

      executeCommands(context->getComputeCommandBuffer(),
                      &(perfParams->elapsedMicrosecs));

      computePerformance({b, h, s, d}, *perfParams, log);
    } else {
      runComputeShader(pipeline, 1, numberOfWorkgroups, 1,
                       context->getComputeCommandBuffer());
      executeCommands(context->getComputeCommandBuffer());
    }

    // Copy matrices to host
    std::vector<std::shared_ptr<DeviceMappedArray>> outputArrays = {
        O, L, S, QT, KtT, ST};
    copyArrays(outputArrays, context->getDeviceToHostTransferCommandBuffer(),
               COPY_DEVICE_TO_HOST);
    executeCommands(context->getDeviceToHostTransferCommandBuffer());

    arrO->copyDataFrom(*O);
#ifdef DEBUG_MODE
    if (buffS != nullptr) {
      std::shared_ptr<HostArray> arrS =
          std::make_shared<HostArray>(buffS, elemSizeS * N * s);
      arrS->copyDataFrom(*S);
    }
#endif
  }

  void computePerformance(std::vector<uint32_t> dims, PerfParams params,
                          std::ostream &log) {

    log << "Done execution. Elapsed microseconds: " << params.elapsedMicrosecs
        << std::endl;

    double tFlop =
        double(params.nReps * 4.0) *
        (double(dims[0] * dims[2] * dims[2]) * double(dims[1] * dims[3])) /
        double(1000.0 * 1000.0 * 1000.0 * 1000.0); // E.g.
    log << "* Total TFLOP: " << tFlop << " (repeats: " << (params.nReps) << ")"
        << std::endl;
    log << "* Total microseconds: " << params.elapsedMicrosecs << std::endl;
    double tFlops =
        (tFlop * (1000.0 * 1000.0)) / double(params.elapsedMicrosecs);
    log << "* Mean kernel execution time: "
        << (double(params.elapsedMicrosecs) / double(params.nReps))
        << std::endl;
    log << "* " << FMT_BEG << "1;32;4m"
        << "TFLOPS: " << tFlops << FMT_CLR << std::endl;
  }

  ~FusedSDPA() {
    vkDestroyPipeline(context->getDevice(), pipeline, NULL);
    vkDestroyShaderModule(context->getDevice(), shaderModule, NULL);
  }
};

template <typename T>
void dummyTorchInterface(void *buffer, uint32_t size, uint32_t dimY,
                         uint32_t dimX) {
  MatrixDesc<T> m =
      MatrixDesc<T>(dimY, dimX, std::make_shared<HostArray>(buffer, size));
  m.printMatrix();
  m.transpose_();
}

typedef struct {
  void *ptr;
  uint32_t size;
} ArrayDesc;

std::shared_ptr<VulkanContext> c = std::make_shared<VulkanContext>();
std::shared_ptr<FusedSDPA> attentionComputePipeline = nullptr;

extern "C" {

void initialize(uint32_t numElems, uint32_t strideB, uint32_t strideH,
                uint32_t strideS, uint32_t strideD, uint32_t elemSizeA,
                uint32_t elemSizeS, uint32_t elemSizeC, float scaling,
                bool causal, bool verbose) {

  std::stringstream ss;
  std::ostream &log = (verbose) ? std::cout : ss;

  attentionComputePipeline = std::shared_ptr<FusedSDPA>(
      new FusedSDPA(c, numElems, {strideB, strideH, strideS, strideD},
                    elemSizeA, elemSizeS, elemSizeC, scaling, causal, log));
}

void callAttention(void *buffQ, void *buffKt, void *buffV, void *buffO,
                   bool verbose) {
  std::stringstream ss;
  std::ostream &log = (verbose) ? std::cout : ss;

  if (attentionComputePipeline == nullptr) {
    throw std::invalid_argument("The attention algorithm should be initialized "
                                "before used by frist calling initialize().");
  }

  // PerfParams pp = {50, 0, 0};
  attentionComputePipeline->call(buffQ, buffKt, buffV, buffO, nullptr,
                                 nullptr, //&pp,
                                 log);
}
}

template <typename InputType = std::float16_t,
          typename IntermediateType = std::float16_t,
          typename AccumulatorType = std::float16_t>
void test(bool causalMasking = false, std::ostream &log = std::cout) {

  uint32_t b = 4;
  uint32_t h = 8;
  uint32_t s = 2048;
  uint32_t d = 64;
  uint32_t N = b * h * s;
  uint32_t numElems = b * h * s * d;

  uint32_t elemSizeA = sizeof(InputType);
  uint32_t elemSizeC = sizeof(AccumulatorType);
  uint32_t elemSizeS = sizeof(IntermediateType);

  bool correctness = false;
  IntermediateType scaling = IntermediateType(1.0);

  attentionComputePipeline = std::shared_ptr<FusedSDPA>(
      new FusedSDPA(c, numElems, {h * s * d, s * d, d, 1}, elemSizeA, elemSizeS,
                    elemSizeC, 1.0, causalMasking, log));

  log << "Testing performance... " << std::endl << std::flush;

  PerfParams pp = {50, 10, 0};

  std::shared_ptr<DeviceMappedArray> Q =
      std::make_shared<DeviceMappedArray>(c, elemSizeA * N * d);
  std::shared_ptr<DeviceMappedArray> K =
      std::make_shared<DeviceMappedArray>(c, elemSizeA * N * d);
  std::shared_ptr<DeviceMappedArray> V =
      std::make_shared<DeviceMappedArray>(c, elemSizeA * N * d);
  std::shared_ptr<DeviceMappedArray> O =
      std::make_shared<DeviceMappedArray>(c, elemSizeC * N * d);
  std::shared_ptr<DeviceMappedArray> L =
      std::make_shared<DeviceMappedArray>(c, elemSizeC * N);
  std::shared_ptr<DeviceMappedArray> S =
      std::make_shared<DeviceMappedArray>(c, elemSizeS * N * s);
  std::shared_ptr<DeviceMappedArray> P =
      std::make_shared<DeviceMappedArray>(c, elemSizeS * N * s);

  MatrixDesc<AccumulatorType> matO = MatrixDesc<AccumulatorType>(N, d, O);
  MatrixDesc<IntermediateType> matS = MatrixDesc<IntermediateType>(N, s, S);

  for (int headId = 0; headId < b * h; headId++) {
    MatrixDesc<InputType> matSliceQ =
        MatrixDesc<InputType>(s, d, Q, headId * s * d * sizeof(InputType));
    MatrixDesc<InputType> matSliceK =
        MatrixDesc<InputType>(s, d, K, headId * s * d * sizeof(InputType));
    MatrixDesc<InputType> matSliceV =
        MatrixDesc<InputType>(s, d, V, headId * s * d * sizeof(InputType));

    matSliceQ.randomInitMatrix_(false);
    matSliceK.randomInitMatrix_(false);
    matSliceV.randomInitMatrix_(false);

    matSliceK.transpose_();

    if (correctness) {
      MatrixDesc<AccumulatorType> matSliceO = MatrixDesc<AccumulatorType>(
          s, d, O, headId * s * d * sizeof(AccumulatorType));
      MatrixDesc<IntermediateType> matSliceS = MatrixDesc<IntermediateType>(
          s, s, S, headId * s * s * sizeof(IntermediateType));
      MatrixDesc<IntermediateType> matSliceP = MatrixDesc<IntermediateType>(
          s, s, P, headId * s * s * sizeof(IntermediateType));
      MatrixDesc<AccumulatorType> matSliceL = MatrixDesc<AccumulatorType>(
          s, 1, L, headId * s * 1 * sizeof(AccumulatorType));

      if (causalMasking) {
        attentionHost<InputType, IntermediateType, AccumulatorType>(
            matSliceQ, matSliceK, matSliceV, matSliceS, matSliceO, matSliceP,
            causalMasking, scaling, log);
      } else {
        attentionHost<InputType, IntermediateType, AccumulatorType>(
            matSliceQ, matSliceK, matSliceV, matSliceS, matSliceO, matSliceP,
            causalMasking, scaling, log);
        // flashAttentionHost(matSliceQ, matSliceK, matSliceV, matSliceO,
        // matSliceL, matSliceS, 16, 16);
      }
    }
  }

  MatrixDesc<IntermediateType> hostMatS = matS.copy();
  MatrixDesc<AccumulatorType> hostMatO = matO.copy();

  attentionComputePipeline->call(Q->getPtr(), K->getPtr(), V->getPtr(),
                                 O->getPtr(), S->getPtr(), &pp, log);

  if (correctness) {
    std::cout << "MSE (S): " << matS.mse(hostMatS) << std::endl;
    hostMatS.printMatrix();
    matS.printMatrix();

    std::cout << "MSE (O): " << matO.mse(hostMatO) << std::endl;
  }
}

void runPerformanceTests() {
  test<std::float16_t, std::float16_t, std::float16_t>();
  test<std::float16_t, std::float16_t, float>();
  test<std::float16_t, float, std::float16_t>();
  test<std::float16_t, float, float>();

  test<std::float16_t, std::float16_t, std::float16_t>(true);
  test<std::float16_t, std::float16_t, float>(true);
  test<std::float16_t, float, std::float16_t>(true);
  test<std::float16_t, float, float>(true);
}

int main(int argc, char *argv[]) {
  // Use the python interface to test the accuracy for larger inputs
  // as it calls PyTorch's attention that runs on GPU.
  runPerformanceTests();

  return 0;
}
