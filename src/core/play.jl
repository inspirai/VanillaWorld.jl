export play!

using REPL
using ReinforcementLearningBase

const ESC = Char(0x1B)
const HIDE_CURSOR = ESC * "[?25l"
const SHOW_CURSOR = ESC * "[?25h"
const CLEAR_SCREEN = ESC * "[2J"
const MOVE_CURSOR_TO_ORIGIN = ESC * "[H"
const CLEAR_SCREEN_BEFORE_CURSOR = ESC * "[1J"
const EMPTY_SCREEN = CLEAR_SCREEN_BEFORE_CURSOR * MOVE_CURSOR_TO_ORIGIN

play!(env::Env) = play!(REPL.TerminalMenus.terminal, env)

function play!(terminal, env)
    out, in = terminal.out_stream, terminal.in_stream
    write(out, CLEAR_SCREEN)
    write(out, MOVE_CURSOR_TO_ORIGIN)
    write(out, HIDE_CURSOR)

    KEYS = action_input_keys(env)

    try
        REPL.Terminals.raw!(terminal, true)
        actions = Int[]
        while true
            write(out, EMPTY_SCREEN)
            show(out, MIME"text/plain"(), env)
            println(out)
            println(out, "Actions:")
            for (x, s) in KEYS
                println(out, "$x: $s")
            end
            println(out, "Q: Quit")
            println(out, "R: Reset")
            last_actions = join((KEYS[a][2] for a in actions), ',')
            println(out, "\nLast Actions: [$(last_actions)]")
            empty!(actions)

            is_reset = false
            for aid in 1:size(action_space(env), 1)
                char = read(in, Char)
                i = findfirst(x -> x[1] == char, KEYS)
                while isnothing(i)
                    if char == 'Q'
                        return
                    elseif char == 'R'
                        reset!(env)
                        is_reset = true
                        break
                    else
                        println(out, "Invalid input!, Try again: ")
                        char = read(in, Char)
                        i = findfirst(x -> x[1] == char, KEYS)
                    end
                end
                if is_reset
                    break
                end
                push!(actions, i)
                println(out, "Action for Agent $aid: $(KEYS[i][2])")
            end
            if !is_reset
                env(reshape(collect(actions), length(actions), 1))
            end
        end
    finally
        write(out, SHOW_CURSOR)
        REPL.Terminals.raw!(terminal, false)
    end
end

#####

export FileRecordingEnv, replay

struct FileRecordingEnv <: AbstractEnv
    env::Env
    io::IO
end

Base.getproperty(env::FileRecordingEnv, x::Symbol) = hasproperty(env, x) ? getfield(env, x) : getproperty(getfield(env, :env), x)

const FRAME_SEPERATOR = "=========================="

function FileRecordingEnv(env::Env, file_name::String)
    io = open(file_name, "w")
    io = IOContext(io, :displaysize=>(2000,3000))
    show(io, MIME"text/plain"(), env)
    FileRecordingEnv(env, io)
end

function (env::FileRecordingEnv)(actions)
    env.env(actions)
    println(env.io, FRAME_SEPERATOR)
    show(env.io, MIME"text/plain"(), env)
    env
end

Base.getindex(env::FileRecordingEnv, i::Int) = FileRecordingEnv(getindex(env, i), env.io)
Base.length(env::FileRecordingEnv) = length(env.env)
device_of(env::FileRecordingEnv) = device_of(env.env)
Adapt.adapt_structure(to, env::FileRecordingEnv) = FileRecordingEnv(Adapt.adapt_structure(to, env.env), env.io)
RLBase.reset!(env::FileRecordingEnv) = reset!(env.env)
RLBase.action_space(env::FileRecordingEnv) = action_space(env.env)
RLBase.state(env::FileRecordingEnv) = state(env.env)
Base.show(io::IO, mime::MIME"text/plain", env::FileRecordingEnv) = show(io, mime, env.env)
Base.close(env::FileRecordingEnv) = close(env.io)

replay(f::String; kw...) = replay(stdout, f; kw...)

function replay(io::IO, f::String; fps=1)
    frames = split(read(f, String), FRAME_SEPERATOR)

    write(io, CLEAR_SCREEN)
    write(io, MOVE_CURSOR_TO_ORIGIN)
    write(io, HIDE_CURSOR)

    try
        for f in frames
            sleep(1 / fps)
            write(io, EMPTY_SCREEN)
            write(io, f)
        end
    finally
        write(io, SHOW_CURSOR)
    end
end