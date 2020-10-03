import Base: getindex, setindex!, size
import LinearAlgebra: transpose, adjoint

using LinearAlgebra: BlasFloat

"""
    ElasticRFP

A resizable representation of triangular, symmetric or hermetian matrices 
based on the rectangular full packed format

`uplo` is assumed to be 'U'
"""
mutable struct ElasticRFP{T<:BlasFloat, V<:DenseMatrix{T}} <: DenseMatrix{T}
    dblock::V # matrix containing diagonal blocks
    oblock::V # matrix containing off-diagonal block
    n::Int
    dA::Char
    capacity::Int  
end
@inline function Base.size(A::ElasticRFP, i) 
    if i == 1 || i == 2
        return A.n
    elseif i > 2
        return 1
    end
    # TODO Throw error
    return 0
end
@inline Base.size(A::ElasticRFP) = size(A, 1), size(A, 2)

@inline function rfpblockdims(A::ElasticRFP)
    isunit = A.dA == 'U'
    c = cld(A.capacity, 2) - (isunit ? 0 : A.capacity % 2)
    a = min(A.n, c)
    b = max(0, A.n - a)
    return a, b
end
@inline function rfpblocks(A::ElasticRFP)
    isunit = A.dA == 'U'
    a, b = rfpblockdims(A)
    c = b + (isunit ? A.capacity % 2 : 0)

    if isunit
        U11t = view(A.dblock, 1:a, 1:a)
    else
        U11t = view(A.dblock, 2:(a+1), 1:a)
    end
    U12 = view(A.oblock, :, 1:b)
    U22 = view(A.dblock, 1:c, 1:c)

    return U11t, U12, U22
end
@inline function Base.getindex(A::ElasticRFP{T}, i::Integer, j::Integer) where T
    isunit = A.dA == 'U'
    if isunit && i == j
        return one(T)
    end
    a, _ = rfpblockdims(A)
    U11t, U12, U22 = rfpblocks(A)
    i, j = min(i,j), max(i,j)
    c = a - (isunit ? A.capacity % 2 : 0)
    if i <= c
        if j <= a
            return U11t[j, i]
        else
            return U12[i, j - a]
        end
    end
    return U22[i - c, j - c]
end
@inline function Base.setindex!(A::ElasticRFP{T}, t::T, i::Integer, j::Integer) where T
    isunit = A.dA == 'U'
    if isunit && i == j && t != one(t)
        # TODO throw error
    end
    a, _ = rfpblockdims(A)
    U11t, U12, U22 = rfpblocks(A)
    i, j = min(i,j), max(i,j)
    c = a - (isunit ? A.capacity % 2 : 0)
    if i <= c
        if j <= a
            U11t[j, i] = t
        else
            U12[i, j - a] = t
        end
    end
    U22[i - c, j - c] = t
    return A
end

"""
"""
struct ElasticSymmetric{T, V<:DenseMatrix{T}} <: DenseMatrix{T}
    rfp::ElasticRFP{T, V}
    tA::Char
end

@inline Base.getindex(A::ElasticSymmetric, i::Integer, j::Integer) = A.tA == 'C' ? conj(A.rfp[i, j]) : A.rfp[i, j]
@inline Base.size(A::ElasticSymmetric, i) = size(A.rfp, i)
@inline Base.size(A::ElasticSymmetric) = size(A.rfp)
    
@inline LinearAlgebra.transpose(A::ElasticSymmetric) = A
@inline function LinearAlgebra.adjoint(A::ElasticSymmetric)
    return ElasticSymmetric(A.rfp,  A.tA == 'C' ? 'N' : 'C')
end

"""
"""
struct ElasticHermitian{T, V<:DenseMatrix{T}} <: DenseMatrix{T}
    rfp::ElasticRFP{T, V}
    uplo::Char
end

@inline function Base.getindex(A::ElasticHermitian, i::Integer, j::Integer)
    return  xor(A.uplo == 'L', i <= j) ? A.rfp[i, j] : conj(A.rfp[i, j])
end
@inline Base.size(A::ElasticHermitian, i) = size(A.rfp, i)
@inline Base.size(A::ElasticHermitian) = size(A.rfp)

@inline function LinearAlgebra.transpose(A::ElasticHermitian) 
    return ElasticHermitian(A.rfp, A.uplo == 'U' ? 'L' : 'U')
end
@inline LinearAlgebra.adjoint(A::ElasticHermitian) = A


"""
"""
struct ElasticTriangular{T, V<:DenseMatrix{T}} <: DenseMatrix{T}
    rfp::ElasticRFP{T, V}
    uplo::Char
    tA::Char
end

@inline function Base.getindex(A::ElasticTriangular{T}, i::Integer, j::Integer) where T
    if i == j
        return A.rfp[i, i]
    elseif xor(!xor(A.uplo == 'L', A.tA == 'C'), i < j)
        return zero(T)
    elseif A.tA == 'C'
        return conj(A.rfp[i, j])
    end
    return A.rfp[i, j]
end
@inline Base.size(A::ElasticTriangular, i) = size(A.rfp, i)
@inline Base.size(A::ElasticTriangular) = size(A.rfp)

@inline function LinearAlgebra.transpose(A::ElasticTriangular) 
    return ElasticTriangular(A.rfp, A.uplo == 'U' ? 'L' : 'U', A.tA)
end
@inline function LinearAlgebra.adjoint(A::ElasticTriangular)
    return ElasticTriangular(A.rfp, A.uplo, A.tA == 'C' ? 'N' : 'C')
end


