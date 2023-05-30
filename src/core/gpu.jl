export cpu, gpu, device_of, device_of

using KernelAbstractions: CPU
using CUDAKernels: CUDADevice
using StructArrays: StructArray, components
using ..Sampler: MySampler

import Adapt
import CUDA

using Random

# copied from Flux.jl

struct CUDAAdaptor end
struct CPUAdaptor end

gpu(x) = CUDA.functional() ? Adapt.adapt(CUDAAdaptor(), x) : x
cpu(x) = Adapt.adapt(CPUAdaptor(), x)

to_device(d, x) = to_device(device_of(d), x)
to_device(::CUDADevice, x) = gpu(x)
to_device(::CPU, x) = cpu(x)

Adapt.adapt_storage(to::CUDAAdaptor, x) = CUDA.cu(x)
Adapt.adapt_storage(to::CPUAdaptor, x::AbstractArray) = Array(x)


device_of(x) = CPU()
device_of(::CUDA.CuArray) = CUDADevice()
device_of(::CUDA.RNG) = CUDADevice()

device_of(x::StructArray) = device_of(components(x))
device_of(x::Tuple) = device_of(first(x))
device_of(x::NamedTuple) = device_of(first(x))

device_of(s::MySampler) = device_of(s.grads)
