export move, towards, distance

function move(I::CartesianIndex{2}, a, UDLR, stride=1)
    if a == UDLR[1]
        I + CartesianIndex(-1, 0) * stride
    elseif a == UDLR[2]
        I + CartesianIndex(1, 0) * stride
    elseif a == UDLR[3]
        I + CartesianIndex(0, -1) * stride
    elseif a == UDLR[4]
        I + CartesianIndex(0, 1) * stride
    else
        nothing # throw error?
    end
end

function towards(src, target)
    dir = sign.(Tuple(target - src))
    src + CartesianIndex(dir)
end

distance(x::CartesianIndex, y::CartesianIndex) = sum(abs.(Tuple(x - y)))

#####

struct2pairs(x) = (k => getfield(x, k) for k in fieldnames(typeof(x)))

function struct2pairs_time(x)
    ks = [:hr_in_day, :min_in_hr]
    vs = [getfield(x, k) for k in ks]
    (ks[i] => vs[i] for i in 1:size(ks, 1))
end

function struct2pairs_convert(x)
    ks = [:role, :action, :health, :emotion, :money, :food, :water, :energy, :enjoyment, :faint, :n_water, :n_grain, :n_meat, :n_roasted_food]
    kstr = ["role", "action", "health", "emotion", "money", "food", "water", "energy", "enjoyment", "faint signal", "water num", "grain num", "meat num", "food num"]
    vs = [getfield(x, k) for k in ks]
    role = vs[1]
    if role == 0
        vs[1] = "Farmer"
    else
        vs[1] = "Hunter"
    end
    (kstr[i] => vs[i] for i in 1:size(ks, 1))
end