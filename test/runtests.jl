using AMXBase
using Test
using SIMD


@testset "Float64 Load and Store" begin
    a = valloc(Float64, 128, 8)
    a .= rand(8)

    b = valloc(Float64, 128, 8)
    b .= 0.0

    AMXBase.amx_set()
    
    gpr_a = AMXBase.ptr_row(UInt64(pointer(a)), 0)
    gpr_b = AMXBase.ptr_row(UInt64(pointer(b)), 0)

    AMXBase.amx_ldx(gpr_a)
    AMXBase.amx_stx(gpr_b)

    @test a == b

    AMXBase.amx_clr()
end

@testset "Float32 Load and Store" begin
    a = valloc(Float32, 128, 16)
    a .= rand(16)

    b = valloc(Float32, 128, 16)
    b .= 0.0

    AMXBase.amx_set()
    
    gpr_a = AMXBase.ptr_row(UInt64(pointer(a)), 0)
    gpr_b = AMXBase.ptr_row(UInt64(pointer(b)), 0)

    AMXBase.amx_ldx(gpr_a)
    AMXBase.amx_stx(gpr_b)

    @test a == b

    AMXBase.amx_clr()
end


@testset "Test FMA64" begin
    a = reshape(valloc(Float64, 128, 64), (8,8))
    a .= reshape(collect(1:64), (8,8))

    b = reshape(valloc(Float64, 128, 64), (8,8))
    b .= reshape(collect(1:64), (8,8))

    c = reshape(valloc(Float64, 128, 4096), (64,64))
    c .= 0.0

    AMXBase.amx_set()
    
    for row in 1:8
        ptr_a = UInt64(pointer(view(a,:,row)))
        gpr_a = AMXBase.ptr_row(ptr_a, row - 1)
        AMXBase.amx_ldx(gpr_a)
        
        ptr_b = UInt64(pointer(view(b,:,row)))
        gpr_b = AMXBase.ptr_row(ptr_b, row - 1)
        AMXBase.amx_ldy(gpr_b)
    end

    AMXBase.amx_fma64(UInt64(0))    

    for row in 1:64
        ptr_c = UInt64(pointer(view(c,:,row)))
        gpr_c = AMXBase.ptr_row(ptr_c, row - 1)
        AMXBase.amx_stz(gpr_c)
    end

    display(a)
    display(b)
    display(c)
    # @test a * b == c'

    AMXBase.amx_clr()
end
