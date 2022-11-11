using LLVM
using LLVM.Interop

@inline function amx_nop_op_imm5(op::UInt32, imm5::UInt32)
  @asmcall("nop")
  @asmcall("nop")
  @asmcall("nop")
  @asmcall(".word (0x201000 + (\$0 << 5) + \$1)", "i,i,~{memory}", Nothing, Tuple{UInt32, UInt32}, op, imm5)
end

@inline function amx_op_gpr(op::UInt32, gpr::UInt64)
  @asmcall(".word (0x201000 + (\$0 << 5) + 0\$1 - ((0\$1 >> 4) * 6))", "i,r,~{memory}", Nothing, Tuple{UInt32, UInt64}, op, gpr)
end


### AMX State

"""
Setup AMX state
*Note:* Raises invalid instruction exception if already setup. All registers set to zero.
"""
@inline amx_set() = amx_nop_op_imm5(UInt32(17), UInt32(0))

"""
Clear AMX state
*Note:* All registers set to uninitialised, no longer need saving/restoring on context switch.
"""
@inline amx_clr() = amx_nop_op_imm5(UInt32(17), UInt32(1))


### Memory Load

"""
Pointer needs to be 128 bit aligned.
x[i] = memory[i]
"""
@inline amx_ldx(gpr) = amx_op_gpr(UInt32(0), gpr)

"""
Pointer needs to be 128 bit aligned.
y[i] = memory[i]
"""
@inline amx_ldy(gpr) = amx_op_gpr(UInt32(1), gpr)

"""
z[i] = memory[i]
"""
@inline amx_ldzr(gpr) = amx_op_gpr(UInt32(4), gpr)

"""
z[_][i] = memory[i]
"""
@inline amx_ldzi(gpr) = amx_op_gpr(UInt32(6), gpr)


### Memory Store

"""
memory[i] = x[i]
"""
@inline amx_stx(gpr) = amx_op_gpr(UInt32(2), gpr)

"""
memory[i] = y[i]
"""
@inline amx_sty(gpr) = amx_op_gpr(UInt32(3), gpr)

"""
memory[i] = z[i]
"""
@inline amx_stz(gpr) = amx_op_gpr(UInt32(5), gpr)

"""
memory[i] = z[_][i]
"""
@inline amx_stzi(gpr) = amx_op_gpr(UInt32(7), gpr)


### Floating-point matrix arithmetic


"""
z[j][i] += x[i] * y[j]
"""
@inline amx_fma64(gpr) = amx_op_gpr(UInt32(10), gpr)


"""
z[j][i] += x[i] * y[j]
"""
@inline amx_fma32(gpr) = amx_op_gpr(UInt32(12), gpr)

"""
z[j][i] += x[i] * y[j]
"""
@inline amx_fma16(gpr) = amx_op_gpr(UInt32(15), gpr)

"""
z[j][i] -= x[i] * y[j]
"""
@inline amx_fms64(gpr) = amx_op_gpr(UInt32(11), gpr)

"""
z[j][i] -= x[i] * y[j]
"""
@inline amx_fms32(gpr) = amx_op_gpr(UInt32(13), gpr)

"""
z[j][i] -= x[i] * y[j]
"""
@inline amx_fms16(gpr) = amx_op_gpr(UInt32(16), gpr)


### Shuffles????

"""
z[j][i] ±= f(x[i], y[j])
"""
@inline amx_matfp(gpr) = amx_op_gpr(UInt32(21), gpr)


### Integer matrix arithmetic

"""
z[j][i] += x[i] * y[j]
"""
@inline amx_mac16(gpr) = amx_op_gpr(UInt32(14), gpr)

"""
z[j][i] ±= f(x[i], y[j])
"""
@inline amx_matint(gpr) = amx_op_gpr(UInt32(20), gpr)


### Floating-point vector arithmetic


### Integer vector arithmetic


### Vector data movement

@inline amx_extrx(gpr) = amx_op_gpr(UInt32(8), gpr)
@inline amx_extry(gpr) = amx_op_gpr(UInt32(9), gpr)


### Vector other

@inline amx_vecint(gpr) = amx_op_gpr(UInt32(18), gpr)
@inline amx_vecfp(gpr) = amx_op_gpr(UInt32(19), gpr)
@inline amx_genlut(gpr) = amx_op_gpr(UInt32(22), gpr)
