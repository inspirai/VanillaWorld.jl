module MA1

using ...Common
using ...Sampler: MySampler

using Random
using ReinforcementLearningBase

using StructArrays: StructArray
using Configurations: @option, from_dict
using YAML: load_file

import CUDA

@enum Role FISHER FARMER COW

#####

struct Tile <: AbstractTile
    id::Int
    agent_id::Int
    npc_id::Int
end

struct Agent <: AbstractAgent
    id::Int
    role::Int
    pos::CartesianIndex{2}

    health::Int
    water::Int
    food::Int
end

Base.convert(::Type{Char}, agent::Agent) = agent.role == Int(FISHER) ? '👨' : '👧'

struct Cow <: AbstractAgent
    id::Int
    role::Int
    pos::CartesianIndex{2}
end

Base.convert(::Type{Char}, x::Cow) = x.role == Int(COW) ? '🐮' : '❓'

struct GlobalState <: AbstractGlobalState
    tick::Int
end

@option struct Config
    seed::Int = 123
    is_use_gpu::Bool = CUDA.functional()
    n_replica::Int = 1
    grid_size::Tuple{Int,Int} = (8, 8)
    n_fisher::Int = 1
    n_farmer::Int = 1
    n_cow::Int = 2
    view_range::Int = 2
    open_simplex_freq::Float32 = 0.1
    water_ratio::Float32 = 0.2
    tree_ratio::Float32 = 0.2
    init_health::Int = 9
    init_water::Int = 9
    init_food::Int = 9

    water_deduction_per_step::Int = 1
    food_deduction_per_step::Int = 1
    health_deduction_per_step::Int = 1
end

function Common.Model(c::Config)
    sampler = MySampler()

    H, W = c.grid_size
    M = c.n_fisher + c.n_farmer
    N = c.n_replica
    V = c.view_range * 2 + 1

    terrain = Terrain(
        StructArray{Tile}(undef, H, W, N),
        (water='🌊', land='🟫', tree='🌲')
    )

    agents = StructArray{Agent}(undef, M, N)
    map!(I -> I[1], agents.id, CartesianIndices(agents.id))
    agents.role[1:c.n_fisher, :] .= Int(FISHER)
    agents.role[c.n_fisher+1:end, :] .= Int(FARMER)

    agents_local_terrain = Terrain(
        StructArray{Tile}(undef, V, V, M, N),
        (water='🌊', land='🟫', tree='🌲')
    )

    cows = StructArray{Cow}(undef, c.n_cow, N)
    map!(I -> I[1], cows.id, CartesianIndices(cows.id))
    cows.role .= Int(COW)

    state = StructArray{GlobalState}(undef, N)

    m = Model(state, terrain, agents, agents_local_terrain, cows, sampler)

    if c.is_use_gpu
        m = gpu(m)
    end

    reset!(m, c)

    m
end

#####
# reset!
#####

function RLBase.reset!(i::Int, t::Terrain, m::Model, c::Config)
    t.agent_id[:, :, i] .= 0
    t.npc_id[:, :, i] .= 0

    n_land_tiles = 0
    n_agents = c.n_farmer + c.n_fisher
    n_npcs = c.n_cow
    H, W, B = size(t)

    seed = rand(Random.default_rng(), UInt)

    for I in CartesianIndices((1:H, 1:W))
        v = (1 + m.sampler(seed, (Tuple(I) .* c.open_simplex_freq)...)) / 2
        if v <= c.water_ratio
            t.id[I, i] = id_of(t, :water)
        else
            if v >= (1 - c.tree_ratio)
                t.id[I, i] = id_of(t, :tree)
            else
                t.id[I, i] = id_of(t, :land)
            end
            n_land_tiles += 1
            # place agents
            if n_land_tiles <= n_agents
                m.agents.pos[n_land_tiles, i] = I
                t.agent_id[I, i] = m.agents.id[n_land_tiles, i]
            else
                x = rand(Random.default_rng(), 1:n_land_tiles)
                if x <= n_agents
                    t.agent_id[m.agents.pos[x, i], i] = 0
                    t.agent_id[I, i] = m.agents.id[x, i]
                    m.agents.pos[x, i] = I
                end
            end
            # place npcs
            if n_land_tiles <= n_npcs
                m.npcs.pos[n_land_tiles, i] = I
                t.npc_id[I, i] = m.npcs.id[n_land_tiles, i]
            else
                x = rand(Random.default_rng(), 1:n_land_tiles)
                if x <= n_npcs
                    t.npc_id[m.npcs.pos[x, i], i] = 0
                    t.npc_id[I, i] = m.npcs.id[x, i]
                    m.npcs.pos[x, i] = I
                end
            end
        end
    end
end

function RLBase.reset!(i::Int, s::StructArray{GlobalState}, m::Model, c::Config)
    s.tick[i] = 1
end

function RLBase.reset!(b::Int, a::StructArray{Agent}, m::Model, c::Config)
    a.health[:, b] .= c.init_health
    a.water[:, b] .= c.init_water
    a.food[:, b] .= c.init_food

    # update local view
    T = m.terrain
    H, W, B = size(T)
    L = m.agents_local_terrain

    for x in 1:size(m.agents, 1)
        p = a.pos[x, b]
        for I in CartesianIndices((-c.view_range:c.view_range, -c.view_range:c.view_range))
            pp = p + I
            ppp = I + CartesianIndex(1 + c.view_range, 1 + c.view_range)
            if pp in CartesianIndices((1:H, 1:W))
                L.id[ppp, x, b] = T.id[pp, b]
                L.agent_id[ppp, x, b] = T.agent_id[pp, b]
                L.npc_id[ppp, x, b] = T.npc_id[pp, b]
            else
                L.id[ppp, x, b] = 0
                L.agent_id[ppp, x, b] = 0
                L.npc_id[ppp, x, b] = 0
            end
        end
    end
end

function RLBase.reset!(i::Int, a::StructArray{Cow}, m::Model, c::Config)
end

#####
# actions
#####

@enum Action begin
    MOVE_LEFT
    MOVE_RIGHT
    MOVE_UP
    MOVE_DOWN
    PICK
    DRINK
end

const UDLR = (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT)
const ACTIONS = instances(Action)

function Common.act!(b::Int, m::Model, c::Config)
    # 1. update global state
    m.state.tick[b] += 1

    # 2. update agent's state
    for i in axes(m.agents, 1)
        m.agents.water[i, b] = clamp(m.agents.water[i, b] - c.water_deduction_per_step, 0, c.init_water)
        m.agents.food[i, b] = clamp(m.agents.food[i, b] - c.food_deduction_per_step, 0, c.init_food)

        if (m.agents.water[i, b] <= 0) || (m.agents.food[i, b] <= 0)
            m.agents.health[i, b] = clamp(m.agents.health[i, b] - c.health_deduction_per_step, 0, c.init_health)
        end
    end

    # 2. update agent's local view
    T = m.terrain
    H, W, B = size(T)
    L = m.agents_local_terrain

    for x in 1:size(m.agents, 1)
        p = m.agents.pos[x, b]
        for I in CartesianIndices((-c.view_range:c.view_range, -c.view_range:c.view_range))
            pp = p + I
            ppp = I + CartesianIndex(1 + c.view_range, 1 + c.view_range)
            if pp in CartesianIndices((1:H, 1:W))
                L.id[ppp, x, b] = T.id[pp, b]
                L.agent_id[ppp, x, b] = T.agent_id[pp, b]
                L.npc_id[ppp, x, b] = T.npc_id[pp, b]
            else
                L.id[ppp, x, b] = 0
                L.agent_id[ppp, x, b] = 0
                L.npc_id[ppp, x, b] = 0
            end
        end
    end
end

function Common.act!(b::Int, m::Model, actions::AbstractMatrix{Int}, c::Config)
    for i in 1:size(m.agents, 1)
        a = ACTIONS[actions[i, b]]
        if a == DRINK
            act_on_drink!(i, b, m, a, c)
        elseif a == PICK
            act_on_pick!(i, b, m, a, c)
        else
            act_on_move!(i, b, m, a, c)
        end
    end
end

function act_on_move!(i::Int, b::Int, m::Model, a, c::Config)
    H, W, B = size(m.terrain)
    src = m.agents.pos[i, b]
    dest = move(src, a, UDLR)

    if !(dest in CartesianIndices((1:H, 1:W))) ||
       (m.terrain.id[dest, b] == id_of(m.terrain, :water)) ||
       (m.terrain.agent_id[dest, b] != 0) ||
       (m.terrain.npc_id[dest, b] != 0)
        dest = src
    end

    m.agents.pos[i, b] = dest
    m.terrain.agent_id[src, b] = 0
    m.terrain.agent_id[dest, b] = i
end

function act_on_drink!(i::Int, b::Int, m::Model, a, c::Config)
    H, W, B = size(m.terrain)
    for I in CartesianIndices((-1:1, -1:1))
        dest = m.agents.pos[i, b] + I
        if dest in CartesianIndices((1:H, 1:W))
            if m.terrain.id[dest, b] == id_of(m.terrain, :water)
                m.agents.water[i, b] += 2
                break
            end
        end
    end
end

function act_on_pick!(i::Int, b::Int, m::Model, a, c::Config)
    H, W, B = size(m.terrain)
    for I in CartesianIndices((-1:1, -1:1))
        dest = m.agents.pos[i, b] + I
        if dest in CartesianIndices((1:H, 1:W))
            if m.terrain.id[dest, b] == id_of(m.terrain, :tree)
                m.agents.food[i, b] += 2
                break
            end
        end
    end
end

#####
# env
#####
export A1

"""
The `A1` environment is the simpliest one provided in this package. It has 3
different tiles: water(🌊), land(🟫), and tree(🌲). The number of agents in this
enviroment is configurable. Each agent has three main attributes: `health`,
`water` and `food`. The `water` and `food` will decrease slowly as time pass by.
Once water or food decreased to `0`, the `health` will drop dramatically. Agents
can move around (up, down, left, righ), drink water to increase the `water`
points, or take in food from trees nearby to increase the `food` points. Our
goal is to keep agents' health in a good state.
"""
function A1(config_file::String="A1.yml"; kw...)
    if isfile(config_file)
        config = load_file(config_file; dicttype=Dict{String,Any})
    else
        config = Dict{String,Any}()
    end
    A1(config; kw...)
end

A1(config::AbstractDict; kw...) = A1(from_dict(Config, config; kw...))

A1(config::Config) = Env(Model(config), config)

RLBase.action_space(env::Env{<:Model,<:Config}) = fill(1:length(ACTIONS), env.config.n_farmer + env.config.n_fisher, env.config.n_replica)

Common.action_input_keys(env::Env{<:Model,<:Config}) = collect(zip("adwsPD", string.(ACTIONS)))

#####
# rendering
#####

function Common.gen_terrain_tile(I, i, terrain, model, config::Config)
    H, W, B = size(terrain)
    V = CartesianIndices((1:H, 1:W))
    L = I + CartesianIndex(0, -1)
    R = I + CartesianIndex(0, 1)
    U = I + CartesianIndex(-1, 0)
    D = I + CartesianIndex(1, 0)
    h, w = Tuple(I)

    tid = terrain.id[I, i]
    if tid == 0
        get_tile(Val(:out_of_map))
    elseif tid == id_of(terrain, :water)
        code = Symbol("water_", (Int(!(x in V) || terrain.id[x, i] == tid) for x in (L, U, R, D))...)
        get_tile(Val(code))
    elseif tid == id_of(terrain, :tree)
        get_tile(Val(:apple))
    elseif tid == id_of(terrain, :land)
        get_tile(Val(:default_background))
    end
end

function Common.gen_agent_tile(I, i, terrain, model, config::Config)
    aid = terrain.agent_id[I, i]
    if aid == 0
        get_tile(Val(:transparent_background))
    else
        role = model.agents.role[aid, i]
        if role == Int(FISHER)
            get_tile(4, 31)
        elseif role == Int(FARMER)
            get_tile(4, 32)
        end
    end
end

function Common.gen_npc_tile(I, i, terrain, model, config::Config)
    npc_id = terrain.npc_id[I, i]
    if npc_id == 0
        get_tile(Val(:transparent_background))
    else
        get_tile(8, 28)
    end
end

end