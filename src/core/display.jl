using Term: Panel, Table, tprint, TextBox, hLine, grid

function readable_pairs(x)
    ks = collect(fieldnames(typeof(x)))
    vs = [getfield(x, k) for k in ks]
    hcat(ks, map(x -> x isa CartesianIndex ? Tuple(x) : x, vs))
end

function readable_pairs_time(x)
    ks = collect(fieldnames(typeof(x)))
    popfirst!(ks)
    vs = [getfield(x, k) for k in ks]
    hcat("Time", string(vs[1], ":", vs[2]))
end

function readable_pairs_minimap(x)
    ks = collect(fieldnames(typeof(x)))
    vs = [getfield(x, k) for k in ks]
    hcat(ks, map(x -> x isa CartesianIndex ? Tuple(x) : x, vs))
end

function readable_pairs_color(x)
    ks = collect(fieldnames(typeof(x)))
    vs = [getfield(x, k) for k in ks]
    hcat(ks, map(x -> x isa CartesianIndex ? Tuple(x) : x, vs))
end

#####
# Terrain
#####

Base.convert(::Type{Array{Char}}, t::Terrain) = map(i -> char_of(t, i), t.id |> cpu)

function _print_symbol_array(io, a::AbstractArray{Char,2})
    for x in eachrow(a)
        for y in x
            print(io, y)
        end
        println(io)
    end
end

Base.show(io::IO, ::MIME"text/plain", t::Terrain) = _print_symbol_array(io, convert(Array{Char}, t))

#####
# Model
#####

function Base.convert(::Type{Array{Char}}, m::Model)
    m = cpu(m)
    A = m.agents
    L = m.agents_local_terrain
    C = m.npcs
    T = m.terrain
    HH, WW, _ = size(m.terrain)
    H, W, N, _ = size(L)
    center = CartesianIndex(ceil(Int, H / 2), ceil(Int, W / 2))

    global_view = convert(Array{Char}, m.terrain)

    for I in CartesianIndices(C)
        global_view[C.pos[I], I[2]] = convert(Char, C[I])
    end

    for I in CartesianIndices(A)
        global_view[A.pos[I], I[2]] = convert(Char, A[I])
    end

    local_view = convert(Array{Char}, L)

    for I in CartesianIndices(local_view)
        h, w, n, b = Tuple(I)
        a_pos = A.pos[n, b]
        shift = CartesianIndex(h, w) - center
        i_pos = a_pos + shift
        if i_pos in CartesianIndices((1:HH, 1:WW))
            if T.agent_id[i_pos, b] != 0
                local_view[I] = convert(Char, A[T.agent_id[i_pos, b], b])
            elseif T.npc_id[i_pos, b] != 0
                local_view[I] = convert(Char, C[T.npc_id[i_pos, b], b])
            end
        end
    end

    global_view, local_view
end

function Base.show(io::IO, mime::MIME"text/plain", m::ModelSlice)
    model = cpu(m.m)
    gv, lv = convert(Array{Char}, model)
    sgv = selectdim(gv, ndims(gv), m.i)
    slv = selectdim(lv, ndims(lv), m.i)

    s_global_view = join((join(r) for r in eachrow(sgv)), "\n")
    main_panel = Panel(
        TextBox(s_global_view, fit=true) / Table(readable_pairs_time(model.state[m.i]), columns_style=["green", "red"], columns_justify=[:right, :left], box=:SIMPLE, compact=true, show_header=false);
        fit=true,
        title="Global View",
        subtitle="$(size(sgv))",
        subtitle_justify=:right,
        style="green"
    )

    # k_v_pair, color_setting = readable_pairs_color(model.agents[i, m.i])
    # local_panel = []
    # for (i, s) in enumerate(eachslice(slv, dims=3))
    #     ks, vs, color_setting = readable_pairs_color(model.agents[i, m.i])

    #     push!(local_panel, Panel(
    #         TextBox(join((join(r) for r in eachrow(s)), "\n"); fit=true) *
    #         Panel(
    #             Table(hcat(ks[1], vs[1]); columns_style=color_setting[1], columns_justify=[:left, :left], box=:SIMPLE, compact=true, show_header=false)/
    #             Table(hcat(ks[2], vs[2]); columns_style=color_setting[2], columns_justify=[:left, :left], box=:SIMPLE, compact=true, show_header=false)/
    #             Table(hcat(ks[3], vs[3]); columns_style=color_setting[2], columns_justify=[:left, :left], box=:SIMPLE, compact=true, show_header=false)/
    #             Table(hcat(ks[4], vs[4]); columns_style=color_setting[2], columns_justify=[:left, :left], box=:SIMPLE, compact=true, show_header=false)/
    #             Table(hcat(ks[5], vs[5]); columns_style=color_setting[2], columns_justify=[:left, :left], box=:SIMPLE, compact=true, show_header=false)/
    #             Table(hcat(ks[6], vs[6]); columns_style=color_setting[2], columns_justify=[:left, :left], box=:SIMPLE, compact=true, show_header=false)/
    #             Table(hcat(ks[7], vs[7]); columns_style=color_setting[2], columns_justify=[:left, :left], box=:SIMPLE, compact=true, show_header=false)/
    #             Table(hcat(ks[8], vs[8]); columns_style=color_setting[2], columns_justify=[:left, :left], box=:SIMPLE, compact=true, show_header=false);
    #             height=20,
    #             width=30,
    #             ); 
    #         fit=true,
    #         style="yellow"
    #     ))
    # end

    local_panel = [
        Panel(
            TextBox(join((join(r) for r in eachrow(s)), "\n"); fit=true) *
            Table(readable_pairs(model.agents[i, m.i]); columns_style=["green", "red"], columns_justify=[:right, :left], box=:SIMPLE, compact=true, show_header=false);
            fit=true,
            style="yellow"
        )
        for (i, s) in enumerate(eachslice(slv, dims=3))
    ]

    p = Panel(
        main_panel * grid(local_panel); # main_panel / grid(local_panel);
        fit=true,
        title="Model on $(nameof(typeof(device_of(m.m))))",
        subtitle="$(m.i)-th frame",
        subtitle_justify=:right,
        style="blue"
    )

    tprint(io, p)
end

function Base.show(io::IO, mime::MIME"text/plain", m::Model)
    gv, lv = convert(Array{Char}, m)
    n = size(gv)[end]
    if n > 0
        gv_i = selectdim(gv, 3, 1)
        _print_symbol_array(io, gv_i)
        lv_i = selectdim(lv, 4, 1)

        for j in 1:size(lv_i, 3)
            println(io)
            _print_symbol_array(io, selectdim(lv_i, 3, j))
        end
    end
end

#####
# Env
#####

function Base.show(io::IO, mime::MIME"text/plain", env::Env)
    n = length(env.model)
    show(io, mime, env.model[1])
    if n > 1
        if n > 2
            println(io)
            println(io, "$(n-2) extra frames are skipped...")
            println(io)
        end
        show(io, mime, env.model[n])
    end
end