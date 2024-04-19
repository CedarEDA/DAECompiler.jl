# This is the pwl_time function from Cedar. It doesn't do anything all that
# special, except that it is a reasonably complicated function that gets lots
# of derivative orders and does not itself have an frule.

using ChainRulesCore
using StaticArrays

struct PWLConstructError
    ts
    ys
end

function Base.showerror(io::IO, err::PWLConstructError)
    println(io, "PWL must have an equal number of x and y values")
end

function find_t_in_ts(ts, t)
    return Base.searchsortedfirst(ts, t)
end

function ChainRulesCore.frule((_, _), ::typeof(find_t_in_ts), ts, t)
    return find_t_in_ts(ts, t), ZeroTangent()
end

rem_right_semi(t, r) = t % r


function ChainRulesCore.frule((_, δt, δr), ::typeof(rem_right_semi), t, r)
    return (rem_right_semi(t, r), δt)
end

@Base.noinline function pwl_at_time(wave, t; r=-1)
    ts = @view wave[1:2:end]
    ys = @view wave[2:2:end]
    if length(ts) != length(ys)
        throw(PWLConstructError(ts, ys))
    end
    if r == 0 # repeat all
        t = rem_right_semi(t, ts[end])
    elseif r > 0 # repeat with period r
        t = rem_right_semi(t, r)
    end
    i = find_t_in_ts(ts, t)
    type_stable_time = 0. * t
    i <= 1 && return ys[1] + type_stable_time
    i > length(ts) && return ys[end] + type_stable_time
    ys[i-1] == ys[i] && return ys[i] + type_stable_time# signal is constant/flat
    ts[i] == ts[i-1] && return (ys[i-1] + ys[i])/2 + type_stable_time # digits truncated to same time values
    frac = (t - ts[i-1]) / (ts[i] - ts[i-1])
    return frac*ys[i] + (1 - frac)*ys[i-1]
end

const example_wave = @SArray[0., 0.,
                             0.5, 0.1,
                             1.0, 0.1,
                             1.5, 0.0]
