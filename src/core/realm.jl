export get_tile, merge_tile

using BlockArrays: BlockArray, Block
using ColorTypes: RGBA, N0f8
using FileIO

const ASSETS_DIR = joinpath(@__DIR__, "..", "assets")

const TILE_SIZE = (32, 32)
const MINIMAL_TILE_SIZE = (16, 16)

const COLORED_1BIT_TILE_PACK_FILE = joinpath(ASSETS_DIR, "colored-transparent_packed.png")
const COLORED_1BIT_TILE_PACK_IMG = load(COLORED_1BIT_TILE_PACK_FILE)
const COLORED_1BIT_TILE_PACK = BlockArray(
    COLORED_1BIT_TILE_PACK_IMG,
    (fill(x, n) for (x, n) in zip(MINIMAL_TILE_SIZE, size(COLORED_1BIT_TILE_PACK_IMG) .÷ MINIMAL_TILE_SIZE))...
)

const FLAG_TILES = Dict(
    splitext(f)[1] => begin
        img = fill(RGBA(0, 0, 0, 0), TILE_SIZE)
        img[17:32, 1:16] .= load(joinpath(ASSETS_DIR, f))
        img
    end
    for f in readdir(ASSETS_DIR) if endswith(f, "flag.png")
)

const DEFAULT_BACKGROUND = fill(RGBA{N0f8}(0.278, 0.176, 0.235, 1.0), TILE_SIZE...)
const TRANSPARENT_BACKGROUND = fill(RGBA{N0f8}(0, 0, 0, 0), TILE_SIZE...)

get_tile(i, j) = repeat(view(COLORED_1BIT_TILE_PACK, Block(i, j)), inner=(2, 2))

get_tile(x::Symbol) = get_tile(Val(x))
get_tile(::Val{:default_background}) = DEFAULT_BACKGROUND
get_tile(::Val{:transparent_background}) = TRANSPARENT_BACKGROUND
get_tile(::Val{:out_of_map}) = get_tile(1, 17)

get_tile(::Val{:water}) = get_tile(6, 9)
get_tile(::Val{:water_1111}) = get_tile(Val(:water))
get_tile(::Val{:water_0111}) = get_tile(6, 10)
get_tile(::Val{:water_1011}) = get_tile(Val(:water_0111)) |> rotr90
get_tile(::Val{:water_1101}) = get_tile(Val(:water_0111)) |> rot180
get_tile(::Val{:water_1110}) = get_tile(Val(:water_0111)) |> rotl90
get_tile(::Val{:water_0011}) = get_tile(5, 10)
get_tile(::Val{:water_1001}) = get_tile(Val(:water_0011)) |> rotr90
get_tile(::Val{:water_1100}) = get_tile(Val(:water_0011)) |> rot180
get_tile(::Val{:water_0110}) = get_tile(Val(:water_0011)) |> rotl90
get_tile(::Val{:water_0101}) = get_tile(5, 9)
get_tile(::Val{:water_1010}) = get_tile(Val(:water_0101)) |> rotr90
get_tile(::Val{:water_0001}) = get_tile(5, 13)
get_tile(::Val{:water_1000}) = get_tile(Val(:water_0001)) |> rotr90
get_tile(::Val{:water_0100}) = get_tile(Val(:water_0001)) |> rot180
get_tile(::Val{:water_0010}) = get_tile(Val(:water_0001)) |> rotl90
get_tile(::Val{:water_0000}) = get_tile(6, 15)

get_tile(::Val{:apple}) = get_tile(19, 34)

get_tile(::Val{:wolf}) = load(joinpath(ASSETS_DIR, "wolf.png"))
get_tile(::Val{X}) where {X} = repeat(load(joinpath(ASSETS_DIR, string(X) * ".png")), inner=(2, 2))

# flags
get_tile(::Val{:attack_flag}) = FLAG_TILES["attack_flag"]
get_tile(::Val{:eat_flag}) = FLAG_TILES["eat_flag"]
get_tile(::Val{:escape_flag}) = FLAG_TILES["escape_flag"]
get_tile(::Val{:collect_water_flag}) = FLAG_TILES["collect_water_flag"]
get_tile(::Val{:cook_flag}) = FLAG_TILES["cook_flag"]
get_tile(::Val{:dance_flag}) = FLAG_TILES["dance_flag"]
get_tile(::Val{:drink_flag}) = FLAG_TILES["drink_flag"]
get_tile(::Val{:farm_flag}) = FLAG_TILES["farm_flag"]
get_tile(::Val{:hi_flag}) = FLAG_TILES["hi_flag"]
get_tile(::Val{:rest_flag}) = FLAG_TILES["rest_flag"]
get_tile(::Val{:sleep_flag}) = FLAG_TILES["sleep_flag"]
get_tile(::Val{:strip_flag}) = FLAG_TILES["strip_flag"]

#####

gen_realm(env::Env, n=nothing) = gen_realm(env.model, env.config, n)
gen_attributes(env::Env, n=nothing) = gen_attributes(env.model, env.config, n)

gen_realm(m::Model, c, n) = gen_realm(m[1], c, n)  # only the first replica by default
gen_attributes(m::Model, c, n) = gen_attributes(m[1], c, n)  # only the first replica by default

gen_realm(model::ModelSlice, config, n) = gen_realm(model.i, model.m, config, n)
gen_attributes(model::ModelSlice, config, n) = gen_attributes(model.i, model.m, config, n)



function gen_realm(i, m::Model, config, ::Nothing)
    H, W, B = size(m.terrain)
    img = BlockArray{RGBA{N0f8}}(undef, fill(TILE_SIZE[1], H), fill(TILE_SIZE[2], W))

    for I in CartesianIndices((1:H, 1:W))
        img[Block(Tuple(I)...)] = gen_tile(I, i, m, config)
    end
    Array(img)
end

function gen_realm(i, m::Model, config, n)
    H, W, B = size(m.agents_local_terrain)
    img = BlockArray{RGBA{N0f8}}(undef, fill(TILE_SIZE[1], H), fill(TILE_SIZE[2], W))

    for I in CartesianIndices((1:H, 1:W))
        img[Block(Tuple(I)...)] = gen_tile(I, i, Terrain(view(m.agents_local_terrain[:tiles], :, :, n, :), m.agents_local_terrain[:tile_meta]), m, config)
    end
    Array(img)
end

function gen_realm_convert(i, m::Model, config, n)
    H, W, B = size(m.agents_local_terrain)
    img = BlockArray{RGBA{N0f8}}(undef, fill(TILE_SIZE[1], H), fill(TILE_SIZE[2], W))

    for I in CartesianIndices((1:H, 1:W))
        img[Block(Tuple(I)...)] = gen_tile(I, i, Terrain(view(m.agents_local_terrain[:tiles], :, :, n, :), m.agents_local_terrain[:tile_meta]), m, config)
    end
    Array(img)
end

gen_tile(I, i, model, config) = gen_tile(I, i, model.terrain, model, config) # global_terrain by default

function gen_tile(I, i, terrain, model, config)
    d = gen_default_tile(I, i, terrain, model, config)
    t = gen_terrain_tile(I, i, terrain, model, config)
    n = gen_npc_tile(I, i, terrain, model, config)
    a = gen_agent_tile(I, i, terrain, model, config)

    merge_tile(d, t, n, a)
end

function merge_tile(args...)
    map(args...) do (args...)
        foldr(args) do x, y
            y.alpha == 0 ? x : y
        end
    end
end

gen_default_tile(I, i, terrain, model, config) = get_tile(Val(:default_background))

function gen_terrain_tile end
function gen_agent_tile end
function gen_npc_tile end

function gen_attributes(i, m::Model, config, ::Nothing)
    struct2pairs_time(m.state[i])
end

function gen_attributes(i, m::Model, config, n)
    # struct2pairs(m.agents[n, i])
    struct2pairs_convert(m.agents[n, i])
end

#####
