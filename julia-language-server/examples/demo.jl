module Demo

export greet

"""Return a greeting for a name."""
function greet(name::AbstractString)
    message = "Hello, " * name
    return message
end

end

using .Demo: greet
println(greet("Julia"))
