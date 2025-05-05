using Test
using SciMLBase
using Sundials
using Compiler
using Core.IR
using DAECompiler
using DAECompiler: get_method_instance, refresh
using DAECompiler.Intrinsics

function get_cached_code_instances(mi::MethodInstance)
  ci = mi.cache
  cached = CodeInstance[ci]
  while isdefined(ci, :next)
    push!(cached, ci.next)
    ci = ci.next
  end
  return cached
end

trivialeq!() = always!(ddt(continuous()))

@testset "Invalidation" begin
  mi = get_method_instance(Tuple{typeof(trivialeq!)}, Base.get_world_counter())

  ci = DAECompiler.find_matching_ci(ci->ci.owner == DAECompiler.StructureCache(), mi, Base.get_world_counter())
  @test ci === nothing

  solve(DAECProblem(trivialeq!, (1,) .=> 1.), IDA())
  ci = DAECompiler.find_matching_ci(ci->ci.owner == DAECompiler.StructureCache(), mi, Base.get_world_counter())
  @test ci !== nothing

  cached = get_cached_code_instances(mi)
  @test getproperty.(cached, :max_world) == fill(typemax(UInt), length(cached))
  world_before = Base.get_world_counter()
  refresh()
  @test getproperty.(cached, :max_world) == fill(world_before, length(cached))

  ci = DAECompiler.find_matching_ci(ci->ci.owner == DAECompiler.StructureCache(), mi, Base.get_world_counter())
  @test ci === nothing

  solve(DAECProblem(trivialeq!, (1,) .=> 1.), IDA())
  ci = DAECompiler.find_matching_ci(ci->ci.owner == DAECompiler.StructureCache(), mi, Base.get_world_counter())
  @test ci !== nothing

  @test length(get_cached_code_instances(mi)) == 2 * length(cached)
end
