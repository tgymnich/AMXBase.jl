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


@testset "Test elemnt wise" begin
    a = reshape(valloc(Float32, 128, 128), (16,8))
    a .= reshape(collect(1:128), (16,8))

    b = reshape(valloc(Float32, 128, 128), (16,8))
    b .= reshape(collect(1:128), (16,8))

    c = reshape(valloc(Float32, 128, 8192), (128,64))
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

    for i in 0:7
        # for j in 0:7
        op = AMXBase.amx_operands_floating_point_arithmetic(64 * i, 64 * i, i, false, false, false, false, 0, false, 0, false, false, false, true)
        AMXBase.amx_fma32(op)
    end

    for row in 1:64
        ptr_c = UInt64(pointer(view(c,:,row)))
        gpr_c = AMXBase.ptr_row(ptr_c, row - 1)
        AMXBase.amx_stz(gpr_c)
    end

    # display(a)
    # display(b)
    # display(a.*b)
    # display(c[1:16,1:8])
    @test a .* b == c[1:16,1:8]

    AMXBase.amx_clr()
end


@testset "Test mat mul" begin
    a = reshape(valloc(Float32, 128, 128), (16,8))
    a .= reshape(collect(1:128), (16,8))

    b = reshape(valloc(Float32, 128, 128), (16,8))
    b .= reshape(collect(1:128), (16,8))

    c = reshape(valloc(Float32, 128, 8192), (128,64))
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

    for i in 0:7
        # for j in 0:7
        op = AMXBase.amx_operands_floating_point_arithmetic(64 * i, 64 * i, i, false, false, false, false, 0, false, 0, false, false, false, true)
        AMXBase.amx_fma32(op)
    end

    for row in 1:64
        ptr_c = UInt64(pointer(view(c,:,row)))
        gpr_c = AMXBase.ptr_row(ptr_c, row - 1)
        AMXBase.amx_stz(gpr_c)
    end

    # display(a)
    # display(b)
    # display(a.*b)
    # display(c[1:16,1:8])
    @test a .* b == c[1:16,1:8]

    AMXBase.amx_clr()
end
