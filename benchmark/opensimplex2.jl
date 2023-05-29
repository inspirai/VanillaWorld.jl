using BenchmarkTools

using CUDA

using VanillaWorld.Sampler: OS2_GRADIENTS_2D, sample

cxs = cu(fill(-1.0f0, 1024, 1024))
grads = cu(OS2_GRADIENTS_2D);

function gpu_os!(xs, freq, start, seed, grads)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(xs)
        pos = Tuple(CartesianIndices(xs)[i] - CartesianIndex(1, 1))
        @inbounds xs[i] = sample((pos .* freq .+ start)...; seed=seed, os2_gradients_2d=grads)
    end
    return nothing
end


function bench_gpu_os!(xs, grads)
    kernel = @cuda launch = false gpu_os!(xs, 1, (1, 1), 123, grads)
    config = launch_configuration(kernel.fun)
    threads = min(length(xs), config.threads)
    blocks = cld(length(xs), threads)
    CUDA.@sync begin
        kernel(xs, 1, (1, 1), 123, grads; threads, blocks)
    end
end

function gen_img!(xs, grads)
    kernel = @cuda launch = false gpu_os!(xs, Float32(2 / 1024), (-1, -1) .+ (2 / 1024), 123, grads)
    config = launch_configuration(kernel.fun)
    threads = min(length(xs), config.threads)
    blocks = cld(length(xs), threads)
    CUDA.@sync begin
        kernel(xs, Float32(2 / 1024), (-1, -1) .+ (2 / 1024), 123, grads; threads, blocks)
    end
end



@btime bench_gpu_os!($cxs, $grads)
# 212.572 μs (10 allocations: 592 bytes)

xs = fill(0.0f0, 1024, 1024)

@btime for i in CartesianIndices($xs)
    $xs[i] = sample(Tuple(i)...)
end
# 26.488 ms (0 allocations: 0 bytes)
