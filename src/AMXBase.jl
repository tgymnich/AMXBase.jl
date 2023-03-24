module AMXBase

const supported_cpus = ["apple-m1", "apple-m2"]

# low-level
include("amx_instructions.jl")
include("amx_operands.jl")

#high-level
include("memory.jl")
include("fma.jl")

end # module AMXBase