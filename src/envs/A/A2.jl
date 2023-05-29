module MA2

using ...Common
using ...Sampler: MySampler

using Random
using ReinforcementLearningBase

using StructArrays: StructArray
using Configurations: @option, from_dict
using YAML: load_file

import CUDA

@enum Role HUNTER CHICKEN WOLF

#####

struct Tile <: AbstractTile
    id::Int
    agent_id::Int
    npc_id::Int
end

struct Hunter <: AbstractAgent
    id::Int
    pos::CartesianIndex{2}
    health::Int
    energy::Int
end

Base.convert(::Type{Char}, agent::Hunter) = '👨'

struct Animal <: AbstractAgent
    id::Int
    role::Int
    health::Int
    pos::CartesianIndex{2}
end

Base.convert(::Type{Char}, x::Animal) = x.role == Int(CHICKEN) ? '🐔' : '🐺'

struct GlobalState <: AbstractGlobalState
    tick::Int
end

@option struct Config
    seed::Int = 123
    is_use_gpu::Bool = CUDA.functional()
    n_replica::Int = 1
    grid_size::Tuple{Int,Int} = (32, 32)
    n_hunter::Int = 8
    n_chicken::Int = 32
    n_wolf::Int = 8

    open_simplex_freq::Float32 = 0.1
    water_ratio::Float32 = 0.1
    rock_ratio::Float32 = 0.1

    init_hunter_health::Int = 10
    init_hunter_energy::Int = 10
    init_chicken_health::Int = 1
    init_wolf_health::Int = 1

    hunter_view_range::Int = 3
    hunter_attack_range::Int = 1
    wolf_view_range::Int = 3
    wolf_attack_range::Int = 1

    hunter_damage::Int = 1
    wolf_damage::Int = 1

    energy_cost_per_fast_move::Int = 1
    energy_recover_per_no_op::Int = 1

    health_recover_per_meat::Int = 11

    hunter_health_deduction_per_tick::Int = 0
end

function Common.Model(c::Config)
    sampler = MySampler()

    H, W = c.grid_size
    M = c.n_hunter
    N = c.n_replica
    V = c.hunter_view_range * 2 + 1

    terrain = Terrain(
        StructArray{Tile}(undef, H, W, N),
        (water='🌊', land='🟫', rock='🗻')
    )

    agents = StructArray{Hunter}(undef, M, N)
    map!(I -> I[1], agents.id, CartesianIndices(agents.id))

    agents_local_terrain = Terrain(
        StructArray{Tile}(undef, V, V, M, N),
        (water='🌊', land='🟫', rock='🗻')
    )

    npcs = StructArray{Animal}(undef, c.n_chicken + c.n_wolf, N)
    map!(I -> I[1], npcs.id, CartesianIndices(npcs.id))
    npcs.role[1:c.n_chicken, :] .= Int(CHICKEN)
    npcs.role[c.n_chicken+1:end, :] .= Int(WOLF)

    state = StructArray{GlobalState}(undef, N)

    m = Model(state, terrain, agents, agents_local_terrain, npcs, sampler)

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
    H, W, B = size(t)

    t.agent_id[:, :, i] .= 0
    t.npc_id[:, :, i] .= 0

    n_land_tiles = 0
    n_agents = c.n_hunter
    n_npcs = c.n_chicken + c.n_wolf

    seed = rand(Random.default_rng(), UInt)

    for I in CartesianIndices((1:H, 1:W))
        v = (1 + m.sampler(seed, (Tuple(I) .* c.open_simplex_freq)...)) / 2
        if v <= c.water_ratio
            t.id[I, i] = id_of(t, :water)
        elseif v >= (1 - c.rock_ratio)
            t.id[I, i] = id_of(t, :rock)
        else
            t.id[I, i] = id_of(t, :land)
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

function update_agents_local_view!(i::Int, m::Model, c::Config)
    A = m.agents
    T = m.terrain
    L = m.agents_local_terrain
    V = c.hunter_view_range
    H, W, B = size(m.terrain)

    for x in 1:size(A, 1)
        p = A.pos[x, i]
        for I in CartesianIndices((-V:V, -V:V))
            pp = p + I
            ppp = I + CartesianIndex(1 + V, 1 + V)
            if pp in CartesianIndices((1:H, 1:W))
                L.id[ppp, x, i] = T.id[pp, i]
                L.agent_id[ppp, x, i] = T.agent_id[pp, i]
                L.npc_id[ppp, x, i] = T.npc_id[pp, i]
            else
                L.id[ppp, x, i] = 0
                L.agent_id[ppp, x, i] = 0
                L.npc_id[ppp, x, i] = 0
            end
        end
    end
end


function RLBase.reset!(i::Int, a::StructArray{Hunter}, m::Model, c::Config)
    a.health[:, i] .= c.init_hunter_health
    a.energy[:, i] .= c.init_hunter_energy

    update_agents_local_view!(i, m, c)
end

function RLBase.reset!(i::Int, a::StructArray{Animal}, m::Model, c::Config)
    for i_npc in 1:size(m.npcs, 1)
        m.npcs.health[i_npc, i] = m.npcs.role[i_npc, i] == Int(CHICKEN) ? c.init_chicken_health : c.init_wolf_health
    end
end

#####
# actions
#####

@enum Action begin
    NO_OP

    MOVE_LEFT
    MOVE_RIGHT
    MOVE_UP
    MOVE_DOWN

    FAST_MOVE_LEFT
    FAST_MOVE_RIGHT
    FAST_MOVE_UP
    FAST_MOVE_DOWN

    PICK
    ATTACK
end

const UDLR = (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT)
const FAST_UDLR = (FAST_MOVE_UP, FAST_MOVE_DOWN, FAST_MOVE_LEFT, FAST_MOVE_RIGHT)
const ACTIONS = instances(Action)

function Common.act!(i::Int, m::Model, c::Config)
    # 1. update global state
    m.state.tick[i] += 1

    # 2. update agent's state
    for aid in axes(m.agents, 1)
        m.agents.health[aid, i] = max(m.agents.health[aid, i] - c.hunter_health_deduction_per_tick, 0)
    end

    update_agents_local_view!(i, m, c)

    # 3. script action
    for npc_id in 1:size(m.npcs, 1)
        if m.npcs.health[npc_id, i] > 0
            if m.npcs.role[npc_id, i] == Int(CHICKEN)
                chicken_act!(npc_id, i, m, c)
            end
            if m.npcs.role[npc_id, i] == Int(WOLF)
                wolf_act!(npc_id, i, m, c)
            end
        end
    end
end

function Common.act!(b::Int, m::Model, actions::AbstractMatrix{Int}, c::Config)
    for i in 1:size(m.agents, 1)
        a = ACTIONS[actions[i, b]]
        if a == ATTACK
            act_on_attack!(i, b, m, a, c)
        elseif a == PICK
            act_on_pick!(i, b, m, a, c)
        elseif a == NO_OP
            m.agents.energy[i, b] = clamp(m.agents.energy[i, b] + c.energy_recover_per_no_op, 0, c.init_hunter_energy)
        else
            act_on_move!(i, b, m, a, c)
        end
    end
end

function act_on_move!(i::Int, b::Int, m::Model, a, c::Config)
    H, W, B = size(m.terrain)
    src = m.agents.pos[i, b]
    is_slow_move = a in UDLR

    if is_slow_move
        dest = move(src, a, UDLR)
    else
        if m.agents.energy[i, b] > 0
            dest = move(src, a, FAST_UDLR, 2)
            m.agents.energy[i, b] = max(m.agents.energy[i, b] - c.energy_cost_per_fast_move, 0)
        else
            dest = move(src, a, UDLR)
        end
    end

    if !(dest in CartesianIndices((1:H, 1:W))) ||
       (m.terrain.id[dest, b] == id_of(m.terrain, :water)) ||
       (m.terrain.id[dest, b] == id_of(m.terrain, :rock)) ||
       (m.terrain.agent_id[dest, b] != 0)
        dest = src
    end

    m.agents.pos[i, b] = dest
    m.terrain.agent_id[src, b] = 0
    m.terrain.agent_id[dest, b] = i
end

function act_on_attack!(aid::Int, i::Int, m::Model, a, c::Config)
    H, W, B = size(m.terrain)
    R = c.hunter_attack_range
    p = m.agents.pos[aid, i]
    for I in CartesianIndices((-R:R, -R:R))
        pp = p + I
        if pp in CartesianIndices((1:H, 1:W))
            npc_id = m.terrain.npc_id[pp, i]
            if npc_id != 0
                m.npcs.health[npc_id, i] = max(0, m.npcs.health[npc_id, i] - c.hunter_damage)
            end
        end
    end
end

function act_on_pick!(aid::Int, i::Int, m::Model, a, c::Config)
    p = m.agents.pos[aid, i]
    npc_id = m.terrain.npc_id[p, i]
    if npc_id != 0 && m.npcs.health[npc_id, i] == 0
        m.agents.health[aid, i] = min(m.agents.health[aid, i] + c.health_recover_per_meat, c.init_hunter_health)
        respawn_npc(npc_id, i, m, c)
    end
end

function respawn_npc(npc_id, i, m, c)
    role = m.npcs.role[npc_id, i]
    p = m.npcs.pos[npc_id, i]
    m.npcs.health[npc_id, i] = role == Int(CHICKEN) ? c.init_chicken_health : c.init_wolf_health

    H, W, B = size(m.terrain)
    while true
        # TODO: limit sampling retry times to H * W
        x = rand(Random.default_rng(), CartesianIndices((1:H, 1:W)))
        xid = m.terrain.npc_id[x, i]
        if xid == 0
            m.npcs.pos[npc_id, i] = x
            m.terrain.npc_id[p, i] = 0
            m.terrain.npc_id[x, i] = npc_id
            break
        end
    end
end

function find_neareast_target(i, center, id_map, max_range, cond)
    # TODO: random sampling
    H, W, B = size(id_map)
    target = CartesianIndex(0, 0)
    is_found = false
    for r in 0:max_range
        a, b = Tuple(center - CartesianIndex(r, r)) # top left
        c, d = Tuple(center + CartesianIndex(r, r)) # bottom right
        for w in (b, d)
            for h in a:c
                p = CartesianIndex(h, w)
                if p in CartesianIndices((1:H, 1:W)) && cond(id_map[p, i])
                    target = p
                    is_found = true
                    break
                end
            end
            is_found && break
        end
        is_found && break
        for h in (a, c)
            for w in b+1:d-1
                p = CartesianIndex(h, w)
                if p in CartesianIndices((1:H, 1:W)) && cond(id_map[p, i])
                    target = p
                    is_found = true
                    break
                end
            end
            is_found && break
        end
        is_found && break
    end
    target
end

function move_npc_to_random_direction(npc_id, i, m, c)
    dir = rand(Random.default_rng(), UDLR)
    src = m.npcs.pos[npc_id, i]
    dest = move(src, dir, UDLR)
    H, W, B = size(m.terrain)

    if !(dest in CartesianIndices((1:H, 1:W))) ||
       (m.terrain.id[dest, i] == id_of(m.terrain, :water)) ||
       (m.terrain.id[dest, i] == id_of(m.terrain, :rock)) ||
       (m.terrain.npc_id[dest, i] != 0)
        dest = src
    end

    m.npcs.pos[npc_id, i] = dest
    m.terrain.npc_id[src, i] = 0
    m.terrain.npc_id[dest, i] = npc_id
end

function wolf_act!(wid, i, m, c)
    wolf_pos = m.npcs.pos[wid, i]
    human_pos = find_neareast_target(i, wolf_pos, m.terrain.agent_id, c.wolf_view_range, >(0))
    if human_pos == CartesianIndex(0, 0)
        chicken_pos = find_neareast_target(i, wolf_pos, m.terrain.npc_id, c.wolf_view_range, x -> 1 <= x <= (c.n_chicken))
        if chicken_pos == CartesianIndex(0, 0)
            move_npc_to_random_direction(wid, i, m, c)
        elseif distance(wolf_pos, chicken_pos) <= c.wolf_attack_range
            chicken_id = m.terrain.npc_id[chicken_pos, i]
            m.npcs.health[chicken_id, i] = max(m.npcs.health[chicken_id, i] - c.wolf_damage, 0)
        else
            dest = towards(wolf_pos, chicken_pos)
            m.npcs.pos[wid, i] = dest
            m.terrain.npc_id[wolf_pos, i] = 0
            m.terrain.npc_id[dest, i] = wid
        end
    elseif distance(wolf_pos, human_pos) <= c.wolf_attack_range
        human_id = m.terrain.agent_id[human_pos, i]
        m.agents.health[human_id, i] = max(m.agents.health[human_id, i] - c.wolf_damage, 0)
    else
        dest = towards(wolf_pos, human_pos)
        m.npcs.pos[wid, i] = dest
        m.terrain.npc_id[wolf_pos, i] = 0
        m.terrain.npc_id[dest, i] = wid
    end
end

function chicken_act!(cid, i, m, c)
    move_npc_to_random_direction(cid, i, m, c)
end

#####
# env
#####
export A2

"""
The `A2` environment is a simple predator and prey game. We have one kind of
agent (the `hunter`) and two kinds of npc (the `wolf` and `chicken`). The wolf
will attack hunter with a higher priority and deal damage to hunter's health.
The hunter can recover health by eating chicken.

The main difference compared to `A1` is that, now resources are moving dynamically.
"""
function A2(config_file::String="A2.yml"; kw...)
    if isfile(config_file)
        config = load_file(config_file; dicttype=Dict{String,Any})
    else
        config = Dict{String,Any}()
    end
    A2(config; kw...)
end

A2(config::AbstractDict; kw...) = A2(from_dict(Config, config; kw...))

A2(config::Config) = Env(Model(config), config)

RLBase.action_space(env::Env{<:Model,<:Config}) = fill(1:length(ACTIONS), env.config.n_hunter, env.config.n_replica)

Common.action_input_keys(env::Env{<:Model,<:Config}) = collect(zip("adwsPD", string.(ACTIONS)))

end