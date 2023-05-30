import .Common as C

using ReinforcementLearningBase
using GLMakie
using Observables

struct MakieEnv{O,I,S}
    env_obs::O
    io::I
    f_save::S
end

export MakieEnv
function MakieEnv(env, path; framerate=4, compression=0, profile="high444", kw...)

    @assert length(env) == 1
    env_obs = Observable(env)
    fig = create_figure(env_obs)
    io = VideoStream(fig; framerate=framerate)
    # f_save = () -> save(path, io; framerate, compression, profile, kw...)
    f_save = () -> save(path, io; framerate=framerate, kw...)
    MakieEnv(env_obs, io, f_save)
end

Base.close(env::MakieEnv) = env.f_save()

function (env::MakieEnv)(actions)
    env.env_obs[](actions)
    env.env_obs[] = env.env_obs[]
    recordframe!(env.io)
    env
end

RLBase.state(env::MakieEnv) = state(env.env_obs[])
RLBase.action_space(env::MakieEnv) = action_space(env.env_obs[])
RLBase.state_space(env::MakieEnv) = state_space(env.env_obs[])

#####

function create_figure(env_obs)
    fig = Figure(resolution=(2400, 1200))
    grid_main = fig[1, 1] = GridLayout()
    create_main_panel(grid_main, env_obs)

    grid_agents = fig[1, 2] = GridLayout()
    create_agents_panel(grid_agents, env_obs)

    colsize!(fig.layout, 1, Relative(1 / 2))

    resize_to_layout!(fig)
    fig
end

function create_main_panel(grid, env_obs)
    grid_terrain = grid[1, 1] = GridLayout()
    ax_terrain = Axis(grid_terrain[1, 1], titlesize=25)
    hidedecorations!(ax_terrain)
    img = @lift(rotr90(C.gen_realm($env_obs)))
    image!(ax_terrain, 0:size(img[], 1), 0:size(img[], 2), img)


    # attrs = @lift(join(["$k => $v" for (k, v) in C.gen_attributes($env_obs)], "\n"))
    attrs = @lift("Time   " * join(["$v " for (k, v) in C.gen_attributes($env_obs)], ": "))
    # attrs = Observable("Time: " * Observables.to_value(attr))
    Label(grid[2, :], attrs, textsize=30)

    colsize!(grid, 1, Aspect(1, 1.0))
    # rowsize!(grid, 2, Relative(1 / 10))
end

function create_agents_panel(grid, env_obs)
    for i in 1:min(2, size(env_obs[].model.agents, 1))
        g = grid[(i-1)%4+1, (i-1)÷4+1] = GridLayout()
        create_sub_panel(g, i, env_obs)
    end
end

function create_sub_panel(g, i, env_obs)
    i_obs = Observable(i + 1)

    ax_terrain = Axis(g[1, 1], title="Local View", titlesize=15)
    hidedecorations!(ax_terrain)
    img = @lift(rotr90(C.gen_realm($env_obs, $i_obs)))
    image!(ax_terrain, 0:size(img[], 1), 0:size(img[], 2), img)

    g_attrs = g[1, 2] = GridLayout()

    attrs_key = @lift(join(["$k" for (k, v) in C.gen_attributes($env_obs, $i_obs)], "\n"))
    Label(g_attrs[2, 1], attrs_key, textsize=24, tellheight=false, justification=:right)

    attrs_val = @lift(join(["$(v isa Number ? round(v;digits=2) : v)" for (k, v) in C.gen_attributes($env_obs, $i_obs)], "\n"))
    Label(g_attrs[2, 2], attrs_val, textsize=24, tellheight=false, justification=:left)

    menu = Menu(g_attrs[1, :], options=[string(x) for x in 1:size(env_obs[].model.agents, 1)], default=string(i), textsize=24)
    on(menu.selection) do s
        i_obs[] = parse(Int, s)
    end

    colsize!(g, 1, Aspect(1, 1.0))
    colsize!(g, 2, Aspect(1, 0.8))
    colsize!(g_attrs, 1, Fixed(100))
end