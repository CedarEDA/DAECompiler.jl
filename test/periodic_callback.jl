using DAECompiler: next_periodic_minimum
using Test

function collect_tstops(list; max_time = Inf, insert_pre_discontinuity_points=false)
    t = 0.0
    tstops = Float64[]
    while t != Inf && t < max_time
        push!(tstops, t)
        tnext = Ref{Float64}(Inf)
        for (offset, period, count) in list
            next_periodic_minimum(tnext, t, offset, period, count, insert_pre_discontinuity_points)
        end
        t = tnext[]
    end
    return tstops
end


# Test basics
four_ticks = [
    (0.1, 1.0, 4),
]
@test collect_tstops(four_ticks) == [0.0, 0.1, 1.1, 2.1, 3.1]

two_double_ticks_then_one = [
    (0.1, 1.0, 2),
    (0.2, 1.0, 2),
    (1.9, Inf, 1),
]
@test collect_tstops(two_double_ticks_then_one) == [0.0, 0.1, 0.2, 1.1, 1.2, 1.9]

# Test edge cases

# Count == -1
infinite_pulse_train = [
    (0.0, 1.0, -1)
]
@test collect_tstops(infinite_pulse_train; max_time=100.0) == Float64.(0:99)

# Count == 0
no_actual_tstops = [
    (1.0, 1.0, 0)
]
@test collect_tstops(no_actual_tstops) == [0.0]

# Count == 1
a_lonely_little_tstop = [
    (1.0, 1.0, 1)
]
@test collect_tstops(a_lonely_little_tstop) == [0.0, 1.0]

# Period < offset
tiny_period = [
    (1.0, 0.1, 4)
]
@test collect_tstops(tiny_period) == [0.0, 1.0, 1.1, 1.2, 1.3]
