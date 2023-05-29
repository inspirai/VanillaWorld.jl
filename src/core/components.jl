export AbstractTile, AbstractAgent, AbstractGlobalState

using ReinforcementLearningBase

#=
Type Herirchy

Env -> Model -> Terrain -> Tile
        |-----> Agent
        |-----> NPC
=#

"""
Each concrete type should contain at least the following fields:

- `id::Int`, usually used to distinguish different terrains.
- `agent_id::Int`, the agent's id on the tile. `0` means nobody.
- `npc_id::Int`, the NPC's id on the tile. `0` means nobody.

To support more agents, developers may consider adding more extra fields.

See also [`Terrain`](@ref).
"""
abstract type AbstractTile end

"""
Each concrete agent implementation should contain at least the following fields:

- `id::Int`, the unique ID of the agent.
- `role::Int`, used to group agents.
- `pos::CartesianIndex{2}`, the agent's position in a 2D terrain.
"""
abstract type AbstractAgent end

"""
The global state is used to track some common data in a game.

Each concrete implementation should at least contain the following fields:

- `tick::Int`, record the number of steps the game has executed.
"""
abstract type AbstractGlobalState end

#####
# Terrain
#####

export Terrain, id_of, char_of

"""
    Terrain(tiles::StructArray{<:AbstractTile}, tile_meta=(tile_name='x', ...))

A wrapper over `StructArray` of a specific `AbstractTile`. `tile_name` will be
used to get the texture of a tile. And the char (`'x'`) will be used to
represent each tile when displayed in terminal.  Note that the `id` field of
each tile ranges between `0` and `length(tile_meta)`.

See also [`id_of`](@ref) and [`char_of`](@ref)
"""
struct Terrain{T,N,A,I<:NamedTuple} <: AbstractArray{T,N}
    tiles::A
    tile_meta::I
end

Terrain(tiles::AbstractArray{T,D}, meta::NamedTuple{names,NTuple{N,Tuple{Int,Char}}}) where {T,D,names,N} = Terrain{T,D,typeof(tiles),typeof(meta)}(tiles, meta)

function Terrain(tiles::A, tile_meta::NamedTuple{names,NTuple{N,Char}}) where {A<:StructArray{<:AbstractTile},names,N}
    T, D = eltype(tiles), ndims(tiles)
    tm = NamedTuple{names}(ntuple(i -> (i, tile_meta[i]), N))
    Terrain{T,D,A,typeof(tm)}(tiles, tm)
end

Base.size(t::Terrain) = size(t[:tiles])
Base.getindex(t::Terrain, s::Symbol) = getfield(t, s)
Base.getproperty(t::Terrain, s::Symbol) = getproperty(t[:tiles], s)
Base.propertynames(t::Terrain) = propertynames(t[:tiles])

"""
    id_of(t::Terrain, s::Symbol, default=0)

Get the id of a tile by its name `s`. If `s` not found in `t`, then `default` is returned.
"""
id_of(t::Terrain, s::Symbol, default=0)::Int = haskey(t[:tile_meta], s) ? t[:tile_meta][s][1] : default

"""
    char_of(t::Terrain, x::Union{Symbol, Int}, default='🛑')

Get the `Char` of a tile by tile name or id.
"""
char_of(t::Terrain, s::Symbol, default='🛑')::Char = haskey(t[:tile_meta], s) ? t[:tile_meta][s][2] : default
char_of(t::Terrain, i::Int, default='🛑')::Char = haskey(t[:tile_meta], i) ? t[:tile_meta][i][2] : default

# to make gpu work on Terrain
Adapt.adapt_structure(to, t::Terrain) = Terrain(Adapt.adapt_structure(to, t[:tiles]), t[:tile_meta])

device_of(t::Terrain) = device_of(t[:tiles])

#####
# Model
#####

export Model, act!
using KernelAbstractions: CPU, @kernel, @index

"""
    Model(args...)

- `state::StructArray{<:AbstractGlobalState, 2}`, record some global info.
- `terrain::Terrain{<:StructArray{<:AbstractTile, 3}}`
- `agents::StructArray{<:AbstractAgent, 2}`
- `agents_local_terrain`, similar to `terrain`, restricted to each agent's current position and view range.
- `npcs::StructArray{<:AbstractAgent}, 2}`
- `sampler`, optional, used to generate terrains dynamically.

Note the last dimension of each field means the number of replicas.
"""
struct Model{S,T,A,L,N,P}
    state::S
    terrain::T
    agents::A
    agents_local_terrain::L
    npcs::N
    sampler::P
end

device_of(m::Model) = device_of(m.state)

Adapt.adapt_structure(to, m::Model) = Model((Adapt.adapt_structure(to, getfield(m, x)) for x in fieldnames(typeof(m)))...)

Base.length(m::Model) = length(m.state)

struct ModelSlice
    m::Model
    i::Int
end

Base.getindex(m::Model, i::Int) = ModelSlice(m, i)

function act!(m::Model, actions::AbstractMatrix{Int}, config)
    device = device_of(m)
    act_kernel!(device, device === CPU() ? Threads.nthreads() : parse(Int, get(ENV, "JULIA_NUM_THREADS_CUDA", "256")))(m, actions, config, ndrange=length(m)) |> wait
end

@kernel function act_kernel!(m::Model, actions::AbstractMatrix{Int}, config)
    b = @index(Global)
    act!(b, m, actions, config)
    act!(b, m, config)
end

function RLBase.reset!(m::Model, c)
    device = device_of(m)
    reset_kernel!(device, device === CPU() ? Threads.nthreads() : parse(Int, get(ENV, "JULIA_NUM_THREADS_CUDA", "256")))(m, c, ndrange=length(m)) |> wait
end

@kernel function reset_kernel!(m::Model, c)
    i = @index(Global)
    reset!(i, m, c)
end

function RLBase.reset!(i::Int, m::Model, c)
    reset!(i, m.terrain, m, c)
    reset!(i, m.agents, m, c)
    reset!(i, m.npcs, m, c)
    reset!(i, m.state, m, c)
end

#####
# env
#####

export Env

struct Env{M,C} <: AbstractEnv
    model::M
    config::C
end

function (env::Env)(actions)
    act!(env.model, to_device(env, actions), env.config)
    env
end

Base.getindex(env::Env, i::Int) = Env(getindex(env.model, i), env.config)
Base.length(env::Env) = length(env.model)

device_of(env::Env) = device_of(env.model)

Adapt.adapt_structure(to, env::Env) = Env(Adapt.adapt_structure(to, env.model), env.config)

RLBase.reset!(env::Env) = reset!(env.model, env.config)

RLBase.state(env::Env) = env.model

#####
# interactivity
#####

export action_input_keys

function action_input_keys end
