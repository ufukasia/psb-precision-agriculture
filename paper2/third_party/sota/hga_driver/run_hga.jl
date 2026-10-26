# Driver for HGSmTSP (Mahmoudinazlou & Kwon 2024), one run.
# usage: julia --project=<env> run_hga.jl <coords.txt> <k> <time_limit_s> <seed> <out.txt>
# coords.txt: line 1 = depot "x y", lines 2..n+1 = points "x y".
# Output: one line per route with 1-based customer ids (point i <-> id i).
using Random, LinearAlgebra
using HGSmTSP

function main()
    inst, k, tl, seed, out = ARGS[1], parse(Int, ARGS[2]), parse(Float64, ARGS[3]), parse(Int, ARGS[4]), ARGS[5]
    rows = [parse.(Float64, split(strip(l))) for l in eachline(inst) if !isempty(strip(l))]
    C = Matrix{Float64}(undef, length(rows), 2)
    for (i, r) in enumerate(rows)
        C[i, 1] = r[1]; C[i, 2] = r[2]
    end
    N = size(C, 1)
    D = Matrix{Float64}(undef, N, N)
    @inbounds for j in 1:N, i in 1:N
        D[i, j] = sqrt((C[i, 1] - C[j, 1])^2 + (C[i, 2] - C[j, 2])^2)
    end
    Random.seed!(seed)
    println("START"); flush(stdout)
    routes, lens = solve_mTSP(k, D, C; time_limit=tl, n_iterations=10^9, verbose=true)
    println("END ", maximum(lens)); flush(stdout)
    open(out, "w") do f
        for r in routes
            println(f, join(r, " "))
        end
    end
end

main()
