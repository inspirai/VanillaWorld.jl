module A

using Reexport

include("A1.jl")
@reexport using .MA1

include("A2.jl")
@reexport using .MA2

end