using ..Common
using ..Sampler: MySampler

using Random
using ReinforcementLearningBase

using StructArrays: StructArray
using Configurations: @option, from_dict
using YAML: load_file

using CUDA: functional

struct Tile <: AbstractTile
    id::Int
    agent_id::Int
    npc_id::Int
    farm_duration::Int
end

const TILE_TYPES = (
    water='🌊',
    land='🟫',
    brick='🟫',
    farmland='🌾',
    farmland_seed='🌰',
    farmland_sprout='🌱',
    flower='🌻', # farmland_grown
    farmland_ripe='🌾',
    fence='🟫',
    campfire='✨',
    tree='🌲',
    shop='🏪',
    bed='🏡'
)

@enum Role FARMER HUNTER CHICKEN WOLF

@enum EntityType AGENT NPC

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
    FARM
    COOK
    COLLECT_WATER
    BUY_GRAIN
    BUY_MEAT
    SELL_GRAIN
    SELL_MEAT
    DRINK
    EAT
    DANCE
    SLEEP
    ATTACK
    STRIP
    SAY_HI
    REST
end

struct Human <: AbstractAgent
    id::Int
    role::Int
    pos::CartesianIndex{2}
    last_action::Action
    action::Action
    bed_pos::CartesianIndex{2}
    farmland_pos::CartesianIndex{2}

    health::Int
    emotion::Int
    money::Int

    food::Float32
    water::Float32
    energy::Float32
    enjoyment::Float32

    faint::Bool

    n_grain::Int
    n_water::Int
    n_meat::Int
    n_roasted_food::Int
    sleep_start_tick::Int
    is_saying_hi::Bool

end

ROLE_DICT = Dict(0 => "Farmer", 1 => "Hunter")
function Common.readable_pairs_minimap(x::Human)
    ks = [:role, :action]
    vs = [getfield(x, k) for k in ks]
end

# function Common.readable_pairs_color(x::Human)
#     # ks = collect(fieldnames(typeof(x)))
#     ks = [:role, :action, :money, :food, :water, :energy, :enjoyment, :faint]
#     need = [:food, :water, :energy, :enjoyment]
#     color_setting = ["green", "red"]
#     vs = []
#     for k in ks
#         v = getfield(x, k)
#         if k in need && v < 30
#             color_setting = ["green", "blue"]
#         end
#         push!(vs, v)
#     end  

#     return hcat(ks, map(x -> x isa CartesianIndex ? Tuple(x) : x, vs)), color_setting
# end

function Common.readable_pairs_color(x::Human)
    # ks = collect(fieldnames(typeof(x)))
    ks = [:role, :action, :money, :food, :water, :energy, :enjoyment, :faint]
    need = [:food, :water, :energy, :enjoyment]
    color_setting = []
    vs = []
    for k in ks
        v = getfield(x, k)
        if k in need && v < 30
            push!(color_setting, ["green", "red"])
        else
            push!(color_setting, ["green", "blue"])
        end
        push!(vs, v)
    end

    role = vs[1]
    if role == 0
        vs[1] = "Farmer"
    else
        vs[1] = "Hunter"
    end

    return ks, vs, color_setting
end

function Common.readable_pairs(x::Human)
    ks = [:role, :action, :money, :food, :water, :energy, :enjoyment, :faint]
    vs = [getfield(x, k) for k in ks]
    role = vs[1]
    if role == 0
        vs[1] = "Farmer"
    else
        vs[1] = "Hunter"
    end
    hcat(ks, map(x -> x isa CartesianIndex ? Tuple(x) : x, vs))
end

Base.convert(::Type{Char}, h::Human) =
    if h.role == Int(FARMER) && h.id == 1
        '👩'
    elseif h.role == Int(FARMER) && h.id == 2
        '🧓'
    elseif h.role == Int(FARMER) && h.id == 3
        '👧'
    elseif h.role == Int(FARMER) && h.id == 4
        '🧒'
    elseif h.role == Int(FARMER)
        '👩'
    elseif h.role == Int(HUNTER) && h.id == 5
        '🧔'
    elseif h.role == Int(HUNTER) && h.id == 6
        '🧑'
    elseif h.role == Int(HUNTER) && h.id == 7
        '👱'
    elseif h.role == Int(HUNTER) && h.id == 8
        '👵'
    elseif h.role == Int(HUNTER)
        '🧔'
    end

struct Animal <: AbstractAgent
    id::Int
    role::Int
    pos::CartesianIndex{2}

    is_attacked::Bool
    health::Int
end

Base.convert(::Type{Char}, a::Animal) =
    if a.role == Int(CHICKEN)
        '🐏'
    elseif a.role == Int(WOLF)
        '🐺'
    end

struct GlobalState <: AbstractGlobalState
    tick::Int
    hr_in_day::Int
    min_in_hr::Int
end

const MINUTES_PER_TICK = 30

# tick starts from the noon (12:00)
hour(tick::Int) = ((tick * MINUTES_PER_TICK + 12 * 60) % (60 * 24)) ÷ 60
mins(tick::Int) = tick % 2 * 30
is_day(tick::Int) = 9 <= hour(tick) <= 17


@option struct Config
    seed::Int = 0
    is_use_gpu::Bool = functional()
    n_replica::Int = 1

    grid_size::Tuple{Int,Int} = (32, 32)
    open_simplex_freq::Float32 = 0.05

    water_ratio::Float32 = 0.25
    illegal_water_range::Int = 9
    tree_ratio::Float32 = 0.1
    flower_ratio::Float32 = 0.22
    brick_ratio::Float32 = 0.0
    farmland_ratio::Float32 = 0.05

    n_shop::Int = 8
    n_farmer::Int = 4
    n_hunter::Int = 4
    n_wolf::Int = 5
    n_chicken::Int = 15

    init_health::Int = 100
    init_emotion::Int = 100
    init_money::Int = 20

    init_food::Int = 144
    init_water::Int = 144
    init_energy::Int = 144
    init_enjoyment::Int = 144

    init_n_grain::Int = 0
    init_n_water::Int = 0
    init_n_meat::Int = 0
    init_n_roasted_food::Int = 0

    init_animal_health::Int = 10
    health_deduction_per_attack::Int = 5

    human_view_range::Int = 3
    camp_range::Int = 7
    legal_farmland_range::Int = 5

    health_recover_per_tick::Int = 10
    starve_threshold::Int = 30
    health_deduction_per_tick_in_starve::Int = 2
    thirsty_threshold::Int = 30
    health_deduction_per_tick_in_thirsty::Int = 2

    food_deduction_per_tick::Float32 = 0.5
    water_deduction_per_tick::Float32 = 0.5
    energy_deduction_per_tick::Float32 = 1
    enjoyment_deduction_per_tick::Float32 = 1

    tired_threshold::Int = 30
    emotion_deduction_per_tick_in_tired::Int = 2
    bored_threshold::Int = 30
    emotion_deduction_per_tick_in_bored::Int = 2

    n_ticks_required_per_farmland_transform::Int = 1
    n_grains_per_harvest::Int = 1
    food_cost_per_cultivate::Int = 5
    water_cost_per_cultivate::Int = 5
    energy_cost_per_cultivate::Int = 5
    enjoyment_cost_per_cultivate::Int = 5
    food_cost_per_attack::Int = 5
    water_cost_per_attack::Int = 5
    energy_cost_per_attack::Int = 5
    enjoyment_cost_per_attack::Int = 5


    human_collect_food_range::Int = 2
    human_collect_water_range::Int = 1

    price_of_grain::Int = 5
    price_of_meat::Int = 20

    food_store_upper_bound::Int = 10
    water_store_upper_bound::Int = 10
    water_increment_per_drink::Int = 70
    food_increment_per_grain::Int = 0
    food_increment_per_meat::Int = 0
    food_increment_per_roasted_food::Int = 70

    n_ticks_required_for_sleep::Int = 1
    energy_increment_per_sleep_in_day::Int = 8
    energy_increment_per_sleep_in_night::Int = 8
    energy_increment_per_rest::Int = 5
    campfire_loc::Tuple{Int,Int} = (12, 12)
    dance_range::Int = 1
    dance_start_time::Int = 16
    dance_end_time::Int = 24
    enjoyment_increment_per_dance::Int = 30
    enjoyment_increasement_on_hi_reacted::Int = 30

    merchant_loc::Tuple{Int,Int} = (20, 20)
    mer_range::Int = 1

    attack_range::Int = 2
    n_meat_provided_by_dead_sheep::Int = 2
    n_meat_provided_by_dead_wolf::Int = 1

    food_cost_per_move::Float32 = 0.4
    water_cost_per_move::Float32 = 0.4
    energy_cost_per_move::Float32 = 0.4
    enjoyment_cost_per_move::Float32 = 0.4

    food_cost_per_fast_move::Int = 1
    water_cost_per_fast_move::Int = 1
    energy_cost_per_fast_move::Int = 1
    enjoyment_cost_per_fast_move::Int = 1

    required_grain_per_cook::Int = 1
    required_meat_per_cook::Int = 1
    acquire_food_per_cook::Int = 1
    acquire_water_per_collect::Int = 1

    say_hi_range::Int = 3

    fast_move_speed::Int = 2

    wolf_view_range::Int = 5
    wolf_attact_range::Int = 1
    wolf_damage::Int = 10
    wolf_pursue_num_per_step::Int = 1
    chicken_noop_ratio::Float32 = 0.7

    money_deduction_per_faint::Int = 20

end

function Common.Model(c::Config)
    sampler = MySampler()

    H, W = c.grid_size
    M = c.n_farmer + c.n_hunter
    C = c.n_chicken + c.n_wolf
    N = c.n_replica
    V = c.human_view_range * 2 + 1

    terrain = Terrain(StructArray{Tile}(undef, H, W, N), TILE_TYPES)

    agents = StructArray{Human}(undef, M, N)
    map!(I -> I[1], agents.id, CartesianIndices(agents.id))
    agents.role[1:c.n_farmer, :] .= Int(FARMER)
    agents.role[c.n_hunter+1:end, :] .= Int(HUNTER)

    agents_local_terrain = Terrain(StructArray{Tile}(undef, V, V, M, N), TILE_TYPES)

    npcs = StructArray{Animal}(undef, C, N)
    map!(I -> I[1], npcs.id, CartesianIndices(npcs.id))
    npcs.role[1:c.n_wolf, :] .= Int(WOLF)
    npcs.role[c.n_wolf+1:end, :] .= Int(CHICKEN)

    state = StructArray{GlobalState}(undef, N)

    m = Model(state, terrain, agents, agents_local_terrain, npcs, sampler)

    if c.is_use_gpu
        m = gpu(m)
    end

    reset!(m, c)

    m
end

function is_within_camp(I::CartesianIndex{2}, c::Config)
    H, W = c.grid_size
    V = c.camp_range
    I in CartesianIndices((H÷2-V:H÷2+V, W÷2-V:W÷2+V))
end

function is_on_bed(I::CartesianIndex{2}, c::Config, m::Model, i::Int)
    # all_res = []
    # for aid in 1:size(m.agents, 1) 
    #     pos =  m.agents.bed_pos[aid,i]
    #     H, W = pos[1],pos[2]
    #     V = c.dance_range
    #     res = I in CartesianIndices((H-V:H+V, W-V:W+V))
    #     push!(all_res,res)
    # end
    # true in all_res
    all_pos = []
    for aid in 1:size(m.agents, 1)
        pos = m.agents.bed_pos[aid, i]
        push!(all_pos, pos)
    end
    I in all_pos
end

function is_illegal_water(I::CartesianIndex{2}, c::Config)
    H, W = c.grid_size
    V = c.illegal_water_range
    I in CartesianIndices((H÷2-V:H÷2+V, W÷2-V:W÷2+V))
end

function is_within_farmland(I::CartesianIndex{2}, c::Config, m::Model, i::Int)
    # H, W = c.grid_size
    # V = c.farmland_range
    # I in CartesianIndices((H÷2-V:H÷2+V, W÷2-V:W÷2+V))
    all_pos = []
    for aid in 1:size(m.agents, 1)
        pos = m.agents.farmland_pos[aid, i]
        push!(all_pos, pos)
    end
    I in all_pos
end

function is_within_campfire(I::CartesianIndex{2}, c::Config)
    H, W = c.campfire_loc
    V = c.dance_range
    I in CartesianIndices((H-V:H+V, W-V:W+V))
end

function is_within_merchant(I::CartesianIndex{2}, c::Config)
    H, W = c.merchant_loc
    V = c.mer_range
    I in CartesianIndices((H-V:H+V, W-V:W+V))
end

function is_on_camp_boarder(I::CartesianIndex{2}, c::Config)
    H, W = c.grid_size
    V = c.camp_range
    (I[1] == H ÷ 2 - V || I[1] == H ÷ 2 + V) && (W ÷ 2 - V <= I[2] <= W ÷ 2 + V) ||
        (H ÷ 2 - V <= I[1] <= H ÷ 2 + V) && (I[2] == W ÷ 2 - V || I[2] == W ÷ 2 + V)
end

function is_on_campfire(I::CartesianIndex{2}, c::Config)
    H, W = c.campfire_loc
    (I[1] == H && I[2] == W)
end

function is_on_merchant(I::CartesianIndex{2}, c::Config)
    H, W = c.merchant_loc
    (I[1] == H && I[2] == W)
end

function RLBase.reset!(i::Int, t::Terrain, m::Model, c::Config)
    t.agent_id[:, :, i] .= 0
    t.npc_id[:, :, i] .= 0
    t.farm_duration[:, :, i] .= 0

    N = c.n_farmer + c.n_hunter
    C = c.n_chicken + c.n_wolf
    H, W, B = size(t)
    seed = rand(Random.default_rng(), UInt)
    seed_tree = rand(Random.default_rng(), UInt)

    n_non_water_tiles = 0
    n_valid_agent_tiles = 0
    n_non_water_tiles_outside_camp = 0

    for I in CartesianIndices((1:H, 1:W))
        v = (1 + m.sampler(seed, (Tuple(I) .* c.open_simplex_freq)...)) / 2
        v_tree = (1 + m.sampler(seed_tree, (Tuple(I) .* c.open_simplex_freq)...)) / 2
        v_flower = rand(Random.default_rng(), 1:1024)
        # add farmland & campfire
        if v <= c.water_ratio
            if is_within_camp(I, c)
                if is_on_campfire(I, c)
                    t.id[I, i] = id_of(t, :campfire)
                elseif is_on_merchant(I, c)
                    t.id[I, i] = id_of(t, :shop)
                elseif is_within_campfire(I, c) || is_within_merchant(I, c)
                    t.id[I, i] = id_of(t, :land)
                else
                    prop = rand(Float64)
                    if prop < c.brick_ratio
                        t.id[I, i] = id_of(t, :brick)
                    else
                        t.id[I, i] = id_of(t, :land)
                    end
                    n_valid_agent_tiles += 1
                end
            else
                if !is_illegal_water(I, c)
                    t.id[I, i] = id_of(t, :water)
                else
                    t.id[I, i] = id_of(t, :land)
                end
            end
        else
            if is_within_camp(I, c)
                # if is_on_camp_boarder(I, c)
                #     t.id[I, i] = id_of(t, :fence)
                # elseif is_within_farmland(I, c)
                #         t.id[I, i] = id_of(t, :farmland)
                if is_on_campfire(I, c)
                    t.id[I, i] = id_of(t, :campfire)
                elseif is_on_merchant(I, c)
                    t.id[I, i] = id_of(t, :shop)
                elseif is_within_campfire(I, c) || is_within_merchant(I, c)
                    t.id[I, i] = id_of(t, :land)
                else

                    prop = rand(Float64)
                    if prop < c.brick_ratio
                        t.id[I, i] = id_of(t, :brick)
                    else
                        t.id[I, i] = id_of(t, :land)
                    end
                    n_valid_agent_tiles += 1
                end
            else
                if v_tree < c.tree_ratio
                    t.id[I, i] = id_of(t, :tree)
                    # elseif v < c.flower_ratio
                    #     t.id[I, i] = id_of(t, :flower)
                else
                    prop = rand(Float64)
                    if prop < c.brick_ratio
                        t.id[I, i] = id_of(t, :brick)
                    else
                        t.id[I, i] = id_of(t, :land)
                    end
                end
            end

            n_non_water_tiles += 1

            # place npcs
            if !is_within_camp(I, c)
                n_non_water_tiles_outside_camp += 1
                if n_non_water_tiles_outside_camp <= C
                    m.npcs.pos[n_non_water_tiles_outside_camp, i] = I
                    t.npc_id[I, i] = m.npcs.id[n_non_water_tiles_outside_camp, i]
                else
                    x = rand(Random.default_rng(), 1:n_non_water_tiles_outside_camp)
                    if x <= C
                        old_pos, new_pos = m.npcs.pos[x, i], I
                        t.npc_id[old_pos, i] = 0
                        t.npc_id[new_pos, i] = m.npcs.id[x, i]
                        m.npcs.pos[x, i] = new_pos
                    end
                end
            end
        end
        # place agents
        if n_valid_agent_tiles >= 1
            if n_valid_agent_tiles <= N
                m.agents.pos[n_valid_agent_tiles, i] = I
                t.agent_id[I, i] = m.agents.id[n_valid_agent_tiles, i]
                m.agents.bed_pos[n_valid_agent_tiles, i] = I
                t.id[I, i] = id_of(t, :bed)
            else
                x = rand(Random.default_rng(), 1:n_valid_agent_tiles)
                if x <= N && !is_within_campfire(I, c) && is_within_camp(I, c) && !is_within_merchant(I, c) && !is_on_camp_boarder(I, c)
                    old_pos, new_pos = m.agents.pos[x, i], I
                    t.agent_id[old_pos, i] = 0
                    t.agent_id[new_pos, i] = m.agents.id[x, i]
                    m.agents.pos[x, i] = new_pos
                    m.agents.bed_pos[x, i] = I
                    t.id[I, i] = id_of(t, :bed)
                    t.id[old_pos, i] = id_of(t, :land)
                end
            end
        end

        # # place agents' bed
        # if I[1] in bed_row && I[2] in bed_col
        #     m.agents.bed_pos[bed_idx, i] = I
        #     t.id[I, i] = id_of(t, :bed)
        #     bed_idx += 1
        # end
    end
    V = c.legal_farmland_range
    all_farmland_pos = []
    for num in 1:size(m.agents, 1)
        row = rand(H÷2-V:H÷2+V)
        col = rand(W÷2-V:W÷2+V)
        I = CartesianIndex(row, col)
        while I in all_farmland_pos || is_within_campfire(I, c) || is_within_merchant(I, c) || is_on_bed(I, c, m, i)
            row = rand(H÷2-V:H÷2+V)
            col = rand(W÷2-V:W÷2+V)
            I = CartesianIndex(row, col)
        end
        push!(all_farmland_pos, I)
        m.agents.farmland_pos[num, i] = I
        t.id[I, i] = id_of(t, :farmland)
    end
end

function RLBase.reset!(i::Int, s::StructArray{GlobalState}, m::Model, c::Config)
    s.tick[i] = 1
    s.hr_in_day[i] = hour(s.tick[i])
    s.min_in_hr[i] = mins(s.tick[i])
end

function RLBase.reset!(i::Int, a::StructArray{Animal}, m::Model, c::Config)
    a.health[:, i] .= c.init_animal_health
    a.is_attacked[:, i] .= false
end

function RLBase.reset!(i::Int, a::StructArray{Human}, m::Model, c::Config)
    a.health[:, i] .= c.init_health
    a.emotion[:, i] .= c.init_emotion
    a.money[:, i] .= c.init_money

    a.food[:, i] .= c.init_food
    a.water[:, i] .= c.init_water
    a.energy[:, i] .= c.init_energy
    a.enjoyment[:, i] .= c.init_enjoyment

    a.n_grain[:, i] .= c.init_n_grain
    a.n_water[:, i] .= c.init_n_water
    a.n_meat[:, i] .= c.init_n_meat
    a.n_roasted_food[:, i] .= c.init_n_roasted_food
    a.is_saying_hi[:, i] .= false
    a.sleep_start_tick[:, i] .= 0
    a.faint[:, i] .= false
    a.action[:, i] .= NO_OP

    a.last_action[:, i] .= NO_OP

    update_agents_local_view!(i, m, c)
end

function update_agents_local_view!(i::Int, m::Model, c::Config)
    A = m.agents
    T = m.terrain
    L = m.agents_local_terrain
    V = c.human_view_range
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
                L.farm_duration[ppp, x, i] = T.farm_duration[pp, i]
            else
                L.id[ppp, x, i] = 0
                L.agent_id[ppp, x, i] = 0
                L.npc_id[ppp, x, i] = 0
                L.farm_duration[ppp, x, i] = 0
            end
        end
    end
end



const UDLR = (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT)
const FAST_UDLR = (FAST_MOVE_UP, FAST_MOVE_DOWN, FAST_MOVE_LEFT, FAST_MOVE_RIGHT)
const ACTIONS = instances(Action)

# TODO:clamp cost here
function Common.act!(i::Int, m::Model, c::Config)
    # 1. global state update
    m.state.tick[i] += 1
    m.state.hr_in_day[i] = hour(m.state.tick[i])
    m.state.min_in_hr[i] = mins(m.state.tick[i])

    # 2. update agents' state
    for aid in 1:size(m.agents, 1)
        # update food
        m.agents.food[aid, i] = clamp(m.agents.food[aid, i] - c.food_deduction_per_tick, 0, c.init_food)
        m.agents.water[aid, i] = clamp(m.agents.water[aid, i] - c.water_deduction_per_tick, 0, c.init_water)
        m.agents.energy[aid, i] = clamp(m.agents.energy[aid, i] - c.energy_deduction_per_tick, 0, c.init_energy)
        m.agents.enjoyment[aid, i] = clamp(m.agents.enjoyment[aid, i] - c.enjoyment_deduction_per_tick, 0, c.init_enjoyment)

        # 2.1 update health
        if m.agents.food[aid, i] > c.starve_threshold && m.agents.water[aid, i] > c.thirsty_threshold
            health_adjustment = c.health_recover_per_tick
        else
            health_adjustment = 0
        end

        if m.agents.food[aid, i] <= c.starve_threshold
            health_adjustment -= c.health_deduction_per_tick_in_starve
        end

        if m.agents.water[aid, i] <= c.thirsty_threshold
            health_adjustment -= c.health_deduction_per_tick_in_thirsty
        end

        m.agents.health[aid, i] = clamp(m.agents.health[aid, i] + health_adjustment, 0, c.init_health)

        # 2.2 update emotion
        if m.agents.energy[aid, i] > c.tired_threshold && m.agents.enjoyment[aid, i] > c.bored_threshold
            emotion_adjustment = c.health_recover_per_tick
        else
            emotion_adjustment = 0
        end
        if m.agents.energy[aid, i] <= c.tired_threshold
            emotion_adjustment -= c.emotion_deduction_per_tick_in_tired
        end

        if m.agents.enjoyment[aid, i] <= c.bored_threshold
            emotion_adjustment -= c.emotion_deduction_per_tick_in_bored
        end

        m.agents.emotion[aid, i] = clamp(m.agents.emotion[aid, i] + emotion_adjustment, 0, c.init_emotion)

        # 2.3 update faint
        m.agents.faint[aid, i] = false
        if m.agents.health[aid, i] <= 0 || m.agents.emotion[aid, i] <= 0
            m.agents.faint[aid, i] = true
            m.agents.health[aid, i] = c.init_health
            m.agents.emotion[aid, i] = c.init_emotion
            m.agents.money[aid, i] = max(m.agents.money[aid, i] - c.money_deduction_per_faint, 0)
            m.agents.food[aid, i] = c.init_food
            m.agents.water[aid, i] = c.init_water
            m.agents.energy[aid, i] = c.init_energy
            m.agents.enjoyment[aid, i] = c.init_enjoyment
        end
    end

    # 3. script action
    for npc_id in 1:size(m.npcs, 1)
        if m.npcs.role[npc_id, i] == Int(CHICKEN)
            chicken_script_act(npc_id, i, m, c)
        end
        if m.npcs.role[npc_id, i] == Int(WOLF)
            wolf_script_act(npc_id, i, m, c)
        end
    end
    update_agents_local_view!(i, m, c)
end

function Common.act!(i::Int, m::Model, actions::AbstractMatrix{Int}, c::Config)
    for aid in 1:size(m.agents, 1)
        a = ACTIONS[actions[aid, i]]
        m.agents.last_action[aid, i] = m.agents.action[aid, i]
        m.agents.action[aid, i] = ACTIONS[actions[aid, i]]
        # reset interact signal if action is not say hi
        if a != SAY_HI
            m.agents.is_saying_hi[aid, i] = false
        end

        if a in UDLR || a in FAST_UDLR
            human_act_on_move!(aid, i, m, a, c)
        elseif a == REST
            human_act_on_rest!(aid, i, m, a, c)
        elseif a == COOK
            human_act_on_cook!(aid, i, m, a, c)
        elseif a == FARM
            human_act_on_farm!(aid, i, m, a, c)
        elseif a == COLLECT_WATER
            human_act_on_collect_water!(aid, i, m, a, c)
        elseif a in (BUY_GRAIN, BUY_MEAT, SELL_GRAIN, SELL_MEAT)
            human_act_on_buy_or_sell!(aid, i, m, a, c)
        elseif a in (DRINK, EAT)
            human_act_on_eat!(aid, i, m, a, c)
        elseif a == SLEEP
            human_act_on_sleep!(aid, i, m, a, c)
        elseif a == DANCE
            human_act_on_dance!(aid, i, m, a, c)
        elseif a == ATTACK
            human_act_on_attack!(aid, i, m, a, c)
        elseif a == STRIP
            human_act_on_strip!(aid, i, m, a, c)
        elseif a == SAY_HI
            human_act_on_say_hi!(aid, i, m, a, c, actions)
        else
            @assert a == NO_OP
        end
    end
end

"""
Fill free time, every step increase energy.
Check:
Nothing
"""
function human_act_on_rest!(aid, i, m, a, c)
    m.agents.energy[aid, i] += c.energy_increment_per_rest
end

"""
Move depend on the speed.
decrease "energy", "water", "food" and "enjoyment"
Check:
1. Cant cross the river.
2. NPC or agent cannot be in the same grid.
"""
function human_act_on_move!(aid, i, m, a, c)
    H, W, B = size(m.terrain)
    src = m.agents.pos[aid, i]
    dest = move(src, a, a in UDLR ? UDLR : FAST_UDLR, a in UDLR ? 1 : c.fast_move_speed)

    if !(dest in CartesianIndices((1:H, 1:W))) ||
       (m.terrain.id[dest, i] == id_of(m.terrain, :water)) ||
       (m.terrain.agent_id[dest, i] != 0) ||
       (m.terrain.npc_id[dest, i] != 0)
        dest = src
    end

    m.agents.pos[aid, i] = dest
    m.terrain.agent_id[src, i] = 0
    m.terrain.agent_id[dest, i] = aid

    # TODO:corner case
    # what if the basic attributes are already at the lowest points?
    # it cant be less than 0
    if a in UDLR
        m.agents.food[aid, i] = max(m.agents.food[aid, i] - c.food_cost_per_move, 0)
        m.agents.water[aid, i] = max(m.agents.water[aid, i] - c.water_cost_per_move, 0)
        m.agents.energy[aid, i] = max(m.agents.energy[aid, i] - c.energy_cost_per_move, 0)
        m.agents.enjoyment[aid, i] = max(m.agents.enjoyment[aid, i] - c.enjoyment_cost_per_move, 0)
    elseif a in FAST_UDLR
        m.agents.food[aid, i] = max(m.agents.food[aid, i] - c.food_cost_per_fast_move, 0)
        m.agents.water[aid, i] = max(m.agents.water[aid, i] - c.water_cost_per_fast_move, 0)
        m.agents.energy[aid, i] = max(m.agents.energy[aid, i] - c.energy_cost_per_fast_move, 0)
        m.agents.enjoyment[aid, i] = max(m.agents.enjoyment[aid, i] - c.enjoyment_cost_per_fast_move, 0)
    else
    end
end

"""
Consume grain and meat, acquire food.
Check:
1. The number of items required is greater than the storage.
2. Storage cant exceed the limit.
"""
function human_act_on_cook!(aid, i, m, a, c)
    if m.agents.n_grain[aid, i] >= c.required_grain_per_cook && m.agents.n_meat[aid, i] >= c.required_meat_per_cook
        m.agents.n_grain[aid, i] -= c.required_grain_per_cook
        m.agents.n_meat[aid, i] -= c.required_meat_per_cook
        m.agents.n_roasted_food[aid, i] += c.acquire_food_per_cook
        m.agents.n_roasted_food[aid, i] = min(m.agents.n_roasted_food[aid, i], c.food_store_upper_bound)
    end
end

"""
collect water.
Check:
1. Water around the agent.
2. Storage cant exceed the limit.
"""
function human_act_on_collect_water!(aid, i, m, a, c)
    R = c.human_collect_water_range
    p = m.agents.pos[aid, i]
    H, W, B = size(m.terrain)
    for I in CartesianIndices((-R:R, -R:R))
        pp = p + I
        if pp in CartesianIndices((1:H, 1:W))
            if m.terrain.id[pp, i] == id_of(m.terrain, :water)
                m.agents.n_water[aid, i] += c.acquire_water_per_collect
                m.agents.n_water[aid, i] = min(m.agents.n_water[aid, i], c.water_store_upper_bound)
                break
            end
        end
    end
end

"""
Acquire grains
decrease "energy", "water", "food" and "enjoyment"
Check:
1. Agent is farmer.
2. Agent is on farmland.
"""
function human_act_on_farm!(aid, i, m, a, c)
    p = m.agents.pos[aid, i]

    if (m.agents.role[aid, i] == Int(FARMER)) &&
       (m.terrain.id[p, i] == id_of(m.terrain, :farmland))
        m.agents.n_grain[aid, i] += c.n_grains_per_harvest
        m.agents.food[aid, i] = max(0, m.agents.food[aid, i] - c.food_cost_per_cultivate)
        m.agents.water[aid, i] = max(0, m.agents.water[aid, i] - c.water_cost_per_cultivate)
        m.agents.energy[aid, i] = max(0, m.agents.energy[aid, i] - c.energy_cost_per_cultivate)
        m.agents.enjoyment[aid, i] = max(0, m.agents.enjoyment[aid, i] - c.enjoyment_cost_per_cultivate)
    end
end

function next_farmland_id(t::Terrain, i::Int)
    if i == id_of(t, :land)
        id_of(t, :farmland)
    elseif i == id_of(t, :farmland)
        id_of(t, :farmland_seed)
    elseif i == id_of(t, :farmland_seed)
        id_of(t, :farmland_sprout)
    elseif i == id_of(t, :farmland_sprout)
        id_of(t, :farmland_grown)
    elseif i == id_of(t, :farmland_grown)
        id_of(t, :farmland_ripe)
    elseif i == id_of(t, :farmland_ripe)
        id_of(t, :farmland)
    else
        0
    end
end

"""
Corresponding items and money change.
Check:
1. Hunter only can buy grain and sell meat. & Farmer only can buy meat and sell grain.
2. money > item price;
3. num store > sell unit;
"""
function human_act_on_buy_or_sell!(aid, i, m, a, c)

    p = m.agents.pos[aid, i]
    R = c.mer_range
    for I in CartesianIndices((-R:R, -R:R))
        pp = p + I
        if m.terrain.id[pp, i] == id_of(m.terrain, :shop)
            if a == SELL_GRAIN && m.agents.n_grain[aid, i] > 0 && m.agents.role[aid, i] == Int(FARMER)
                m.agents.n_grain[aid, i] -= 1
                m.agents.money[aid, i] += c.price_of_grain
            elseif a == SELL_MEAT && m.agents.n_meat[aid, i] > 0 && m.agents.role[aid, i] == Int(HUNTER)
                m.agents.n_meat[aid, i] -= 1
                m.agents.money[aid, i] += c.price_of_meat
            elseif a == BUY_GRAIN && m.agents.money[aid, i] >= c.price_of_grain && m.agents.role[aid, i] == Int(HUNTER)
                m.agents.n_grain[aid, i] += 1
                m.agents.money[aid, i] -= c.price_of_grain
            elseif a == BUY_MEAT && m.agents.money[aid, i] >= c.price_of_meat && m.agents.role[aid, i] == Int(FARMER)
                m.agents.n_meat[aid, i] += 1
                m.agents.money[aid, i] -= c.price_of_meat
            else
            end
        end
    end
end

"""
Consume food/water and recover "food"/ "water".
Check:
1. item store > 0;
"""
function human_act_on_eat!(aid, i, m, a, c)
    if a == DRINK && m.agents.n_water[aid, i] > 0
        m.agents.n_water[aid, i] -= 1
        m.agents.water[aid, i] += c.water_increment_per_drink
        # elseif a == EAT_GRAIN && m.agents.n_grain[aid, i] > 0
        #     m.agents.n_grain[aid, i] -= 1
        #     m.agents.food[aid, i] += c.food_increment_per_grain
        # elseif a == EAT_MEAT && m.agents.n_meat[aid, i] > 0
        #     m.agents.n_meat[aid, i] -= 1
        #     m.agents.food[aid, i] += c.food_increment_per_meat
    elseif a == EAT && m.agents.n_roasted_food[aid, i] > 0
        m.agents.n_roasted_food[aid, i] -= 1
        m.agents.food[aid, i] += c.food_increment_per_roasted_food
    else
    end
end

"""
Recover "energy".
Check:
1. sleep legal pos
"""
function human_act_on_sleep!(aid, i, m, a, c)
    # if m.agents.sleep_start_tick[aid, i] == 0  # default value
    #     m.agents.sleep_start_tick[aid, i] = m.state.tick[i]
    # end
    if m.agents.pos[aid, i] == m.agents.bed_pos[aid, i]
        m.agents.energy[aid, i] += c.energy_increment_per_sleep_in_night
    end
    # if m.state.tick[i] - m.agents.sleep_start_tick[aid, i] >= c.n_ticks_required_for_sleep
    #     if is_day(m.state.tick[i])
    #         m.agents.energy[aid, i] += c.energy_increment_per_sleep_in_day
    #     else
    #         m.agents.energy[aid, i] += c.energy_increment_per_sleep_in_night
    #     end

    #     m.agents.sleep_start_tick[aid, i] = m.state.tick[i] # otherwise, following up sleep will always increase energy
    # end
end

"""
Recover "enjoyment".
Check:
1. Campfire around the agent.
2. Legal time.
"""
function human_act_on_dance!(aid, i, m, a, c)
    R = c.camp_range
    H, W, B = size(m.terrain)
    p = m.agents.pos[aid, i]
    for I in CartesianIndices((-R:R, -R:R))
        pp = p + I
        if pp in CartesianIndices((1:H, 1:W))
            if m.terrain.id[pp, i] == id_of(m.terrain, :campfire) && (c.dance_start_time <= hour(m.state.tick[i]) <= c.dance_end_time)
                m.agents.enjoyment[aid, i] += c.enjoyment_increment_per_dance
                break
            end
        end
    end

end

"""
The initiator needs to wait for other agents to respond.
If recive a respond, both increase "enjoyment".
Check:
1. There is other NPC in range.
2. Others also choose the "SAY_HI" action.
"""
function human_act_on_say_hi!(aid, i, m, a, c, actions)
    m.agents.is_saying_hi[aid, i] = true
    src_pos = m.agents.pos[aid, i]
    R = c.say_hi_range

    for xid in 1:size(m.agents, 1)
        if xid == aid
            continue
        end
        dest_pos = m.agents.pos[xid, i]
        if (src_pos - dest_pos in CartesianIndices((-R:R, -R:R))) &&
           (m.agents.is_saying_hi[xid, i] == true || ACTIONS[actions[xid, i]] == SAY_HI)
            m.agents.enjoyment[aid, i] += c.enjoyment_increasement_on_hi_reacted
            break
        end
    end
end

"""
Attack the chicken to acquire the meat.
Check:
1. Char: hunter.
2. Target: chicken.
3. Chicken in the attack range.
"""
function human_act_on_attack!(aid, i, m, a, c)
    R = c.attack_range
    H, W, B = size(m.terrain)
    N = size(m.agents, 1)

    if m.agents.role[aid, i] == Int(HUNTER)
        p = m.agents.pos[aid, i]
        for I in CartesianIndices((-R:R, -R:R))
            pp = p + I
            if pp in CartesianIndices((1:H, 1:W))
                x = m.terrain.npc_id[pp, i]
                if x > 0 && (m.npcs.role[x, i] == Int(CHICKEN) || m.npcs.role[x, i] == Int(WOLF))
                    m.npcs.health[x, i] = clamp(m.npcs.health[x, i] - c.health_deduction_per_attack, 0, c.init_animal_health)
                    m.agents.food[aid, i] = max(0, m.agents.food[aid, i] - c.food_cost_per_attack)
                    m.agents.water[aid, i] = max(0, m.agents.water[aid, i] - c.water_cost_per_attack)
                    m.agents.energy[aid, i] = max(0, m.agents.energy[aid, i] - c.energy_cost_per_attack)
                    m.agents.enjoyment[aid, i] = max(0, m.agents.enjoyment[aid, i] - c.enjoyment_cost_per_attack)
                end
            end
        end
    end
end

"""
Strip from the dead npc, and randomly place a new animal.
Check:
1. Char: hunter.
2. Exists dead npc around hunter.
"""
function human_act_on_strip!(aid, i, m, a, c)
    R = c.human_collect_food_range
    H, W, B = size(m.terrain)
    N = size(m.agents, 1)
    if m.agents.role[aid, i] == Int(HUNTER)
        p = m.agents.pos[aid, i]
        for I in CartesianIndices((-R:R, -R:R))
            pp = p + I
            if pp in CartesianIndices((1:H, 1:W))
                x = m.terrain.npc_id[pp, i]
                if x > 0 && m.npcs.health[x, i] == 0
                    if m.npcs.role[x, i] == Int(CHICKEN)
                        m.agents.n_meat[aid, i] += c.n_meat_provided_by_dead_sheep
                    elseif m.npcs.role[x, i] == Int(WOLF)
                        m.agents.n_meat[aid, i] += c.n_meat_provided_by_dead_wolf
                    end
                    # then randomly place this animal somewhere
                    m.terrain.npc_id[pp, i] = 0
                    num_t = 0
                    while true
                        new_p = rand(Random.default_rng(), CartesianIndices((1:H, 1:W)))
                        num_t += 1
                        if m.terrain.npc_id[new_p, i] == 0 && !is_within_camp(new_p, c) && m.terrain.id[new_p, i] == 2
                            m.terrain.npc_id[new_p, i] = m.npcs.id[x, i]
                            m.npcs.pos[x, i] = new_p
                            m.npcs.health[x, i] = c.init_animal_health
                            break
                        end
                        if num_t > 20
                            new_p = pp
                            m.terrain.npc_id[new_p, i] = m.npcs.id[x, i]
                            m.npcs.pos[x, i] = new_p
                            m.npcs.health[x, i] = c.init_animal_health
                            break
                        end
                    end
                end
            end
        end
    end
end

# script npc

function chicken_script_act(npc_id, i, m, c)
    #spawn
    if m.npcs.health[npc_id, i] <= 0
        # npc_spawn(npc_id, i, m, c)
        return
    end
    #move


    if m.npcs.is_attacked[npc_id, i]
        move_npc_random_direction(npc_id, i, m, c)
        move_npc_random_direction(npc_id, i, m, c)
        m.npcs.is_attacked[npc_id, i] = false
    else
        prop = rand(Float64)
        if prop > c.chicken_noop_ratio
            move_npc_random_direction(npc_id, i, m, c)
        end
    end
end

function wolf_script_act(npc_id, i, m, c)
    wolf_pos = m.npcs.pos[npc_id, i]
    human_pos = find_neareast_human(i, wolf_pos, m.terrain, c.wolf_view_range)
    if human_pos == CartesianIndex(0, 0) || is_within_campfire(human_pos, c) || is_within_merchant(human_pos, c) || is_on_bed(human_pos, c, m, i)
        move_npc_random_direction(npc_id, i, m, c)
    else
        pursue_cnt = 0
        while pursue_cnt < c.wolf_pursue_num_per_step
            if distance(wolf_pos, human_pos) <= c.wolf_attact_range
                human_id = m.terrain.agent_id[human_pos, i]
                m.agents.health[human_id, i] -= c.wolf_damage
                return
            end
            dest = wolf_pos + _toward(wolf_pos, human_pos)
            if is_place_pos_valid(NPC, dest, i, m, c)
                m.npcs.pos[npc_id, i] = dest
                m.terrain.npc_id[wolf_pos, i] = 0
                m.terrain.npc_id[dest, i] = npc_id
            end
            pursue_cnt += 1
        end
    end
end

#script utils

function is_place_pos_valid(EntityType, pos, i, m, c)
    H, W, B = size(m.terrain)

    if !(pos in CartesianIndices((1:H, 1:W))) ||
       (m.terrain.id[pos, i] == id_of(m.terrain, :water))
        return false
    end

    if EntityType == AGENT && m.terrain.agent_id[pos, i] != 0
        return false
    end

    if EntityType == NPC && (is_within_campfire(pos, c) || is_within_merchant(pos, c) || is_on_bed(pos, c, m, i) || is_within_farmland(pos, c, m, i) || m.terrain.npc_id[pos, i] != 0)
        return false
    end

    return true
end

function get_npc_spawn_pos(i, m, c)
    H, W, B = size(m.terrain)
    cnt = 0

    pos = CartesianIndex(rand(Random.default_rng(), 1:H), rand(Random.default_rng(), 1:W))
    while !is_place_pos_valid(NPC, pos, i, m, c) && cnt < 30
        pos = CartesianIndex(rand(Random.default_rng(), 1:H), rand(Random.default_rng(), 1:W))
        cnt += 1
    end
    return pos
end

function npc_spawn(npc_id, i, m, c)
    pos = get_npc_spawn_pos(i, m, c)
    if !is_place_pos_valid(NPC, pos, i, m, c)
        return
    end
    old_pos = m.npcs.pos[npc_id, i]
    m.terrain.npc_id[old_pos, i] = 0
    m.terrain.npc_id[pos, i] = npc_id
    m.npcs.pos[npc_id, i] = pos
    m.npcs.health[npc_id, i] = c.init_animal_health
    m.npcs.health[npc_id, i] = false
end

function move_npc_random_direction(npc_id, i, m, c)
    dir = rand(Random.default_rng(), UDLR)
    npc_pos = m.npcs.pos[npc_id, i]
    dest = move(npc_pos, dir, UDLR)
    H, W, B = size(m.terrain)

    if !is_place_pos_valid(NPC, dest, i, m, c)
        dest = npc_pos
    end

    m.npcs.pos[npc_id, i] = dest
    m.terrain.npc_id[npc_pos, i] = 0
    m.terrain.npc_id[dest, i] = npc_id
end

function find_neareast_human(i, center, terrain, max_range)
    # TODO: random sampling
    H, W, B = size(terrain)
    human = CartesianIndex(0, 0)
    is_found = false
    for r in 0:max_range
        a, b = Tuple(center - CartesianIndex(r, r)) # top left
        c, d = Tuple(center + CartesianIndex(r, r)) # bottom right
        for w in (b, d)
            for h in a:c
                p = CartesianIndex(h, w)
                if p in CartesianIndices((1:H, 1:W)) && terrain.agent_id[p, i] != 0
                    human = p
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
                if p in CartesianIndices((1:H, 1:W)) && terrain.agent_id[h, w, i] != 0
                    human = p
                    is_found = true
                    break
                end
            end
            is_found && break
        end
        is_found && break
    end
    human
end

function _toward(pos, target_pos)
    diff = target_pos - pos
    dists = abs.(Tuple(diff))
    if dists[1] > dists[2]
        return CartesianIndex(sign(diff[1]), 0)
    else
        return CartesianIndex(0, sign(diff[2]))
    end
end

#####
# env
#####
export W2

function W2(config_file::String="W2.yml"; kw...)
    if isfile(config_file)
        config = load_file(config_file; dicttype=Dict{String,Any})
    else
        config = Dict{String,Any}()
    end
    W2(config; kw...)
end

W2(config::AbstractDict; kw...) = W2(from_dict(Config, config; kw...))

W2(config::Config) = Env(Model(config), config)

RLBase.action_space(env::Env{<:Model,<:Config}) = fill(1:length(ACTIONS), env.config.n_farmer + env.config.n_hunter, env.config.n_replica)

Common.action_input_keys(env::Env{<:Model,<:Config}) = collect(zip("0adwsADWS123456789zxcvbnm", string.(ACTIONS)))

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
    elseif tid == id_of(terrain, :land)
        get_tile(Val(:default_background))
    elseif tid == id_of(terrain, :brick)
        get_tile(3, 9)
    elseif tid == id_of(terrain, :farmland)
        get_tile(7, 17)
    elseif tid == id_of(terrain, :farmland_seed)
        get_tile(1, 7)
    elseif tid == id_of(terrain, :farmland_sprout)
        get_tile(3, 1)
    elseif tid == id_of(terrain, :flower)
        get_tile(3, 2)
    elseif tid == id_of(terrain, :farmland_ripe)
        get_tile(3, 5)
    elseif tid == id_of(terrain, :fence)
        get_tile(Val(:default_background))
    elseif tid == id_of(terrain, :campfire)
        get_tile(11, 15)
    elseif tid == id_of(terrain, :tree)
        get_tile(2, 2)
    elseif tid == id_of(terrain, :shop)
        get_tile(21, 8)
    elseif tid == id_of(terrain, :bed)
        get_tile(20, 1)
    end
end

function Common.gen_agent_tile(I, i, terrain, model, config::Config)
    aid = terrain.agent_id[I, i]
    if aid == 0
        get_tile(Val(:transparent_background))
    else
        role = model.agents.role[aid, i]
        id = model.agents.id[aid, i]

        agent_tile = if role == Int(HUNTER) && id == 5
            get_tile(:hunter_1)
        elseif role == Int(HUNTER) && id == 6
            get_tile(:hunter_2)
        elseif role == Int(HUNTER) && id == 7
            get_tile(:hunter_3)
        elseif role == Int(HUNTER) && id == 8
            get_tile(:hunter_4)
        elseif role == Int(FARMER) && id == 1
            get_tile(:farmer_1)
        elseif role == Int(FARMER) && id == 2
            get_tile(:farmer_2)
        elseif role == Int(FARMER) && id == 3
            get_tile(:farmer_3)
        elseif role == Int(FARMER) && id == 4
            get_tile(:farmer_4)
        end

        action = model.agents.action[aid, i]
        action_tile = if action == FARM
            get_tile(:farm_flag)
        elseif action == COOK
            get_tile(:cook_flag)
        elseif action == COLLECT_WATER
            get_tile(:collect_water_flag)
        elseif action == DRINK
            get_tile(:drink_flag)
        elseif action == EAT
            get_tile(:eat_flag)
        elseif action == DANCE
            get_tile(:dance_flag)
        elseif action == SLEEP
            get_tile(:sleep_flag)
        elseif action == ATTACK
            get_tile(:attack_flag)
        elseif action == STRIP
            get_tile(:strip_flag)
        elseif action == SAY_HI
            get_tile(:hi_flag)
        elseif action == REST
            get_tile(:rest_flag)
        else
            get_tile(:transparent_background)
        end

        merge_tile(agent_tile, action_tile)
    end
end

function Common.gen_npc_tile(I, i, terrain, model, config::Config)
    npc_id = terrain.npc_id[I, i]
    if npc_id == 0
        get_tile(Val(:transparent_background))
    else
        role = model.npcs.role[npc_id, i]
        if role == Int(CHICKEN)
            get_tile(:chicken)
        else
            get_tile(:wolf)
        end
    end
end
