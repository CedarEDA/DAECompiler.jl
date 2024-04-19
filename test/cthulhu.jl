module cthulhu

using Test
using DAECompiler, SciMLBase
using REPL, Cthulhu

include(joinpath(Base.pkgdir(DAECompiler), "test", "lorenz.jl"))

module FakeTerminal

using REPL: REPL
using Test

export fake_terminal, keydict

function fake_terminal(func; options::REPL.Options=REPL.Options(confirm_exit=false))
    # Use pipes so we can easily do blocking reads
    # In the future if we want we can add a test that the right object
    # gets displayed by intercepting the display
    input = Pipe()
    output = Pipe()
    err = Pipe()
    Base.link_pipe!(input, reader_supports_async=true, writer_supports_async=true)
    Base.link_pipe!(output, reader_supports_async=true, writer_supports_async=true)
    Base.link_pipe!(err, reader_supports_async=true, writer_supports_async=true)

    term_env = get(ENV, "TERM", @static Sys.iswindows() ? "" : "dumb")
    term = REPL.Terminals.TTYTerminal(term_env, input.out, IOContext(output.in, :color=>get(stdout, :color, false)), err.in)
    out = func(term, input.in, output.out, err)
    t = @async begin
        close(input.in)
        close(output.in)
        close(err.in)
    end
    @test isempty(read(err.out, String))
    wait(t)
    return out
end

const keydict = Dict(
    :up => "\e[A",
    :down => "\e[B",
    :enter => '\r')

end # module FakeTerminal
using .FakeTerminal

# TODO interrupt the failed task automatically
macro redirect_to_stderr(tryex)
    :(try
        $(esc(tryex))
    catch err
        bt = catch_backtrace()
        Base.display_error(stderr, err, bt)
    end)
end

let u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 100.0)
    x = Lorenz1ts(10.0, 28.0, 8.0/3.0)
    sys = IRODESystem(Tuple{typeof(x)};
        debug_config=(;store_ir_levels=true,))
    daeprob = DAEProblem(sys, zero(u0), u0, tspan, x)
    @test fake_terminal() do terminal, in, out, err
        t = @async @redirect_to_stderr descend(daeprob; terminal)
        readuntil(out, DAECompiler.SELECT_IR_MSG; keep=true)
        write(in, keydict[:enter])     # enter into the Cthulhu view
        readuntil(out, '↩'; keep=true) # read until '↩' at the last of the Cthulhu view
        write(in, 'q')                 # back to the select ir view
        readuntil(out, DAECompiler.SELECT_IR_MSG; keep=true)
        write(in, keydict[:down])      # choose another IR
        write(in, keydict[:enter])     # enter into the Cthulhu view
        readuntil(out, '↩'; keep=true) # read until '↩' at the last of the Cthulhu view
        write(in, 'q')                 # back to the select ir view
        write(in, 'q')                 # quit
        wait(t)
        return true
    end
end

end # module cthulhu
