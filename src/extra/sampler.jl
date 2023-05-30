module Sampler

# This file is modified from https://github.com/mfiano/CoherentNoise.jl/blob/main/src/noise/opensimplex2_noise.jl
# to support random generation on GPU
# See more discussions here: https://github.com/mfiano/CoherentNoise.jl/issues/2

using CUDA
using FastPow
import Adapt

const HASH_MULTIPLIER = 0x53a3f72deec546f5
const PRIME_X = 0x5205402b9270c86f
const PRIME_Y = 0x598cd327003817b5

const OS2_SKEW_2D = 0.366025403784439f0
const OS2_UNSKEW_2D = -0.21132486540518713f0
const OS2_R²2D = 0.5f0
const OS2_NUM_GRADIENTS_EXP_2D = 7
const OS2_NUM_GRADIENTS_2D = 1 << OS2_NUM_GRADIENTS_EXP_2D
const OS2_GRADIENTS_NORMALIZED_2D = Float32[
    0.38268343236509, 0.923879532511287,
    0.923879532511287, 0.38268343236509,
    0.923879532511287, -0.38268343236509,
    0.38268343236509, -0.923879532511287,
    -0.38268343236509, -0.923879532511287,
    -0.923879532511287, -0.38268343236509,
    -0.923879532511287, 0.38268343236509,
    -0.38268343236509, 0.923879532511287,
    0.130526192220052, 0.99144486137381,
    0.608761429008721, 0.793353340291235,
    0.793353340291235, 0.608761429008721,
    0.99144486137381, 0.130526192220051,
    0.99144486137381, -0.130526192220051,
    0.793353340291235, -0.60876142900872,
    0.608761429008721, -0.793353340291235,
    0.130526192220052, -0.99144486137381,
    -0.130526192220052, -0.99144486137381,
    -0.608761429008721, -0.793353340291235,
    -0.793353340291235, -0.608761429008721,
    -0.99144486137381, -0.130526192220052,
    -0.99144486137381, 0.130526192220051,
    -0.793353340291235, 0.608761429008721,
    -0.608761429008721, 0.793353340291235,
    -0.130526192220052, 0.99144486137381]
const OS2_GRADIENTS_2D = OS2_GRADIENTS_NORMALIZED_2D ./ 0.01001634121365712f0

"""
    opensimplex2_2d(; kwargs...)
Construct a sampler that outputs 2-dimensional OpenSimplex2 noise when it is sampled from.

# Arguments
  - `seed=0`: An integer used to seed the random number generator for this sampler.
  - `orient=nothing`: Either the symbol `:x` or the value `nothing`:
      + `:x`: The noise space will be re-oriented with the Y axis pointing down the main diagonal to
        improve visual isotropy.
      + `nothing`: Use the standard orientation.
"""
opensimplex2_2d(; seed=0, orient=nothing) = opensimplex2(2, seed, orient)

@inline function grad(table, seed, X, Y, x, y)
    N = length(table)
    hash = (seed ⊻ X ⊻ Y) * HASH_MULTIPLIER
    hash ⊻= hash >> (64 - OS2_NUM_GRADIENTS_EXP_2D + 1)
    i = trunc(hash) & ((OS2_NUM_GRADIENTS_2D - 1) << 1)
    t = (table[mod1(i + 1, N)], table[mod1((i | 1) + 1, N)])
    sum((t .* (x, y)))
end

# @inline transform(::OpenSimplex2{2,OrientStandard}, x, y) = (x, y) .+ OS2_SKEW_2D .* (x + y)
@inline transform(x, y) = (x, y) .+ OS2_SKEW_2D .* (x + y)

# @inline function transform(::OpenSimplex2{2,OrientX}, x, y)
#     xx = x * ROOT_2_OVER_2
#     yy = y * ROOT_2_OVER_2 * (2OS2_SKEW_2D + 1)
#     (yy + xx, yy - xx)
# end

@fastpow function sample(x::T, y::T; seed=123, os2_gradients_2d=OS2_GRADIENTS_2D) where {T<:Real}
    primes = (PRIME_X, PRIME_Y)
    tr = transform(x, y)
    XY = floor.(Int, tr)
    vtr = tr .- XY
    t = sum(vtr) * OS2_UNSKEW_2D
    X1, Y1 = XY .* primes
    X2, Y2 = (X1, Y1) .+ primes
    x1, y1 = vtr .+ t
    us1 = 2OS2_UNSKEW_2D + 1
    result = 0.0f0
    a1 = OS2_R²2D - x1^2 - y1^2
    if a1 > 0
        result += a1^4 * grad(os2_gradients_2d, seed, X1, Y1, x1, y1)
    end
    a2 = 2us1 * (1 / OS2_UNSKEW_2D + 2) * t + -2us1^2 + a1
    if a2 > 0
        x, y = (x1, y1) .- 2OS2_UNSKEW_2D .- 1
        result += a2^4 * grad(os2_gradients_2d, seed, X2, Y2, x, y)
    end
    if y1 > x1
        x = x1 - OS2_UNSKEW_2D
        y = y1 - OS2_UNSKEW_2D - 1
        a3 = OS2_R²2D - x^2 - y^2
        if a3 > 0
            result += a3^4 * grad(os2_gradients_2d, seed, X1, Y2, x, y)
        end
    else
        x = x1 - OS2_UNSKEW_2D - 1
        y = y1 - OS2_UNSKEW_2D
        a4 = OS2_R²2D - x^2 - y^2
        if a4 > 0
            result += a4^4 * grad(os2_gradients_2d, seed, X2, Y1, x, y)
        end
    end
    result
end

#####

export MySampler

Base.@kwdef struct MySampler{T}
    grads::T = OS2_GRADIENTS_2D
end

Adapt.adapt_structure(to, s::MySampler) = MySampler(Adapt.adapt_structure(to, s.grads))

function (s::MySampler)(seed, h, w)
    sample(h, w; seed=seed, os2_gradients_2d=s.grads)
end

end