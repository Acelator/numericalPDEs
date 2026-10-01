# 2D Poisson solver with finite differences on (0,1)^2
# 5-point stencil, ordering (i, j) -> u_ji
# Matrix covers interior unknowns only (Dirichlet boundaries go to RHS)

#  -laplacian(u) = f in (0,1)^2
#   sample for testing: f(x,y) = (2*pi^2 * sin(pi*x)*sin(pi*y))
#       with null Dirichlet condition


using SparseArrays
using LinearAlgebra

# All the code assumes that Nx = Ny
Nx = 100 # Points in x
Ny = 100 # Points in y
NxInt = Nx - 2
NyInt = Ny - 2

@assert Nx == Ny && NxInt == NyInt

N = NxInt * NyInt # interior unknowns

h = 1.0 / (NxInt + 1) # uniform mesh spacing

# Source function
function f(x::Float64, y::Float64)
    return 2 * pi^2 * sin(pi*x) * sin(pi*y)
end

# Diricihlet boundary
function g(x::Float64, y::Float64)
    return 0
end


# Kronecker assembly
function laplacian2d_kron(n::Int, h::Float64)
    e = ones(Float64, n)

    # 1D Dirichlet Laplacian
    D = spdiagm(
        -1 => -e[1:(n-1)],
        0 => 2.0 .* e,
        1 => -e[1:(n-1)])
    In = sparse(1.0I, n, n)

    # 2D Laplacian as Kronecker sum
    return (kron(In, D) + kron(D, In)) / (h^2)
end

# COO assembly, interior only
function laplacian2d_construction(n::Int, h::Float64)
    idx(i, j) = (i - 1) * n + j
    Nloc = n * n

    # Preallocate storage
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]
    sizehint!(I_idx, 5 * Nloc)
    sizehint!(J_idx, 5 * Nloc)
    sizehint!(V_val, 5 * Nloc)

    k = 1.0 / (h^2)

    for i in 1:n
        for j in 1:n
            row = idx(i, j)

            push!(I_idx, row)
            push!(J_idx, row)
            push!(V_val, 4.0 * k)

            if i > 1
                push!(I_idx, row)
                push!(J_idx, idx(i - 1, j))
                push!(V_val, -k)
            end

            if i < n
                push!(I_idx, row)
                push!(J_idx, idx(i + 1, j))
                push!(V_val, -k)
            end

            if j > 1
                push!(I_idx, row)
                push!(J_idx, idx(i, j - 1))
                push!(V_val, -k)
            end

            if j < n
                push!(I_idx, row)
                push!(J_idx, idx(i, j + 1))
                push!(V_val, -k)
            end
        end
    end

    return sparse(I_idx, J_idx, V_val, Nloc, Nloc)
end

# f in the source function and g the one that defines the Dirichlet condition
function assemble_rhs(n::Int, h::Float64, f::Function, g::Function)
    b = zeros(Float64, n * n)
    idx(i, j) = (i - 1) * n + j
    k = 1.0 / (h^2)

    for j in 1:n
        y = j * h
        for i in 1:n
            x = i * h
            pos = idx(i, j)

            b[pos] = f(x, y)

            # Dirichlet boundary conditions
            if i == 1       # Left border (x = 0)
                b[pos] += k * g(0.0, y)
            end
            if i == n       # Right border (x = 1)
                b[pos] += k * g(1.0, y)
            end
            if j == 1       # Below border (y = 0)
                b[pos] += k * g(x, 0.0)
            end
            if j == n       # Above border (y = 1)
                b[pos] += k * g(x, 1.0)
            end
        end
    end

    return b
end

# Kronecker assembly is way faster and also use less peak memory.
A_kron = laplacian2d_kron(NxInt, h)
A_coo = laplacian2d_construction(NxInt, h)

# Both assembly routes must agree
@assert A_kron ≈ A_coo

# Sparsity stats
println("Size: ", size(A_kron))
println("nnz: ", nnz(A_kron))
println("Zeros: ", 100 * (1 - nnz(A_kron) / (N^2)), "%")

# b = assemble_rhs(NxInt, h, f, g)
# Q = cholesky(A_kron)
# xInt = Q \ b  # Value in the interior points   
# print(x)


function uExact(x::Float64, y::Float64)
    return sin(pi*x) * sin(pi*y)
end

function compute_error(n::Int, h::Float64, u)
    u_exact = zeros(Float64, n * n)
    idx(i, j) = (i-1)*n + j

    for j in 1:n
        y = j * h
        for i in 1:n
            x = i * h
            u_exact[idx(i, j)] = uExact(x, y)
        end
    end

    # Obtain numerical error
    error = u .- u_exact

    # L-Infinity norm
    err_inf = norm(error, Inf)

    # L-2 norm
    err_L2 = h * norm(error, 2)

    # println("Error in L-infinity norm: ", err_inf)
    # println("Error in L-2 norm: ", err_L2)

    return err_L2
end

# compute_error(NxInt, h, xInt)

println("--------------------------------")
newError::Float64 = 0
lastError::Float64 = 0
# Points in each region
for i in [50, 100, 200, 400]

    NInt = i - 2

    local N = NInt * NInt # interior unknowns

    local h = 1.0 / (NInt + 1) # uniform mesh spacing

    local A_kron = laplacian2d_kron(NInt, h)

    b = assemble_rhs(NInt, h, f, g)
    Q = cholesky(A_kron)
    xInt = Q \ b  # Value in the interior points   

    global newError = compute_error(NInt, h, xInt)
    if lastError == 0
        println("Initial error with N=", NInt, " is:", newError)
    else
        cocient = - log2(newError / lastError)
        println("Cocient of errors with N=", NInt, " is:", cocient)
    end
    global lastError = newError

    # We see that in fact the method has order 2
end

# Full vector including Dirichlet boundary, same row-major order
function assemble_full(nTot::Int, h::Float64, xInt::AbstractVector, g::Function)
    n = nTot - 2 # interior per side
    xFull = zeros(Float64, nTot * nTot)
    idxFull(it, jt) = (it - 1) * nTot + jt
    idxInt(i, j) = (i - 1) * n + j

    for jt in 1:nTot
        y = (jt - 1) * h
        for it in 1:nTot
            x = (it - 1) * h
            if it == 1 || it == nTot || jt == 1 || jt == nTot
                xFull[idxFull(it, jt)] = g(x, y) # boundary value
            else
                xFull[idxFull(it, jt)] = xInt[idxInt(it - 1, jt - 1)] # interior copy
            end
        end
    end
    return xFull
end