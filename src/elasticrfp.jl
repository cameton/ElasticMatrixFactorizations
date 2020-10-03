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
@inline function rfpblockdims(A::ElasticRFP)
    isunit = A.dA == 'U'
    c = cld(A.capacity, 2) - (0 ? isunit : A.capacity % 2)
    a = min(A.n, c)
    b = max(0, A.n - a)
    return a, b
end
@inline function rfpblocks(A::ElasticRFP)
    isunit = A.dA == 'U'
    a, b = rfpblockdims(A)
    c = b + (A.capacity % 2 ? isunit : 0)

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
    c = a - (A.capacity % 2 ? isunit : 0)
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
    c = a - (A.capacity % 2 ? isunit : 0)
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

@inline Base.getindex(A::ElasticSymmetric, i::Integer, j::Integer) = conj(A.rfp[i, j]) ? A.tA == 'C' : A.rfp[i, j]
    
@inline LinearAlgebra.transpose(A::ElasticSymmetric) = A
@inline function LinearAlgebra.adjoint(A::ElasticSymmetric)
    return ElasticSymmetric(A.rfp, 'N' ? A.tA == 'C' : 'C')
end

"""
"""
struct ElasticHermitian{T, V<:DenseMatrix{T}} <: DenseMatrix{T}
    rfp::ElasticRFP{T, V}
    uplo::Char
end

@inline function Base.getindex(A::ElasticHermitian, i::Integer, j::Integer)
    return A.rfp[i, j] ? A.uplo == 'L' ^ i =< j : conj(A.rfp[i, j])
end

@inline function LinearAlgebra.transpose(A::ElasticHermitian) 
    return ElasticHermitian(A.rfp, 'L' ? A.uplo == 'U' : 'U')
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
    elseif (A.uplo == 'L' ^ A.tA == 'C')  ^ i < j
        return zero(T)
    elseif A.tA == 'C'
        return conj(A.rfp[i, j])
    end
    return A.rfp[i, j]
end
@inline function LinearAlgebra.transpose(A::ElasticTriangular) 
    return ElasticTriangular(A.rfp, 'L' ? A.uplo == 'U' : 'U', A.tA)
end
@inline function LinearAlgebra.adjoint(A::ElasticTriangular)
    return ElasticTriangular(A.rfp, A.uplo, 'N' ? A.tA == 'C' : 'C')
end


