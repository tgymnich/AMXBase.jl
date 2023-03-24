using BitOperations

@enum RegisterCount begin
    one
    two
    four # M2 only
end


export ldx, ldy, ldz, ldzi, stx, sty, stz, stzi

"""
Load pair
x[i] = memory[i]

# Arguments
- `pointer::UInt64`: the pointer.
- `register_index::Integer`: X register index.
- `register_count::RegisterCount`: Either one, two or four.
"""
@inline function ldx(pointer::UInt64; register_index::Integer=0, register_count::RegisterCount=one)
    two_registers = register_count != one
    four_registers = register_count == four
    
    return amx_ldx(amx_operands_memory_xy(pointer, register_index, four_registers, two_registers))
end


"""
Load pair
y[i] = memory[i]

# Arguments
- `pointer::UInt64`: the pointer.
- `register_index::Integer`: Y register index.
- `register_count::RegisterCount`: Either one, two or four.
"""
@inline function ldy(pointer::UInt64; register_index::Integer=0, register_count::RegisterCount=one)
    two_registers = register_count != one
    four_registers = register_count == four
    
    return amx_ldy(amx_operands_memory_xy(pointer, register_index, four_registers, two_registers))
end


"""
Load pair
z[i] = memory[i]

# Arguments
- `pointer::UInt64`: the pointer.
- `register_index::Integer`: Z row.
- `double_width::Bool`: Load / store pair of registers (1) or single register (0).
"""
@inline function ldz(pointer::UInt64; register_index::Integer=0, double_width::Bool=false)   
    return amx_ldz(amx_operands_memory_z(pointer, register_index, double_width))
end


"""
Load pair, interleaved Z
z[_][i] = memory[i]

# Arguments
- `pointer::UInt64`: the pointer.
- `register_index::Integer`: Z row (high 5 bits thereof).
- `right_hand_half::Bool`: Right hand half (1) or left hand half (0) of Z register pair.
"""
@inline function ldzi(pointer::UInt64; register_index::Integer=0, right_hand_half::Bool=false)  
    return amx_ldzi(amx_operands_memory_zi(pointer, right_hand_half, register_index))
end



### Store Instruction


"""
Store pair
memory[i] = x[i]

# Arguments
- `pointer::UInt64`: the pointer.
- `register_index::Integer`: X register index.
- `double_width::RegisterCount`: Store pair of registers (1) or single register (0).
"""
@inline function stx(pointer::UInt64; register_index::Integer=0, double_width::Bool=false)
    return amx_stx(amx_operands_memory_xy(pointer, register_index, false, double_width))
end


"""
Store pair
memory[i] = y[i]

# Arguments
- `pointer::UInt64`: the pointer.
- `register_index::Integer`: Y register index.
- `double_width::RegisterCount`: Store pair of registers (1) or single register (0).
"""
@inline function sty(pointer::UInt64; register_index::Integer=0, double_width::Bool=false)
    amx_sty(amx_operands_memory_xy(pointer, register_index, false, double_width))
end


"""
Store pair
memory[i] = z[i]

# Arguments
- `pointer::UInt64`: the pointer.
- `register_index::Integer`: Y register index.
- `register_count::RegisterCount`: Either one, two or four
"""
@inline function stz(pointer::UInt64; register_index::Integer=0, double_width::Bool=false)   
    amx_stz(amx_operands_memory_z(pointer, register_index, double_width))
end


"""
Store pair, interleaved Z
memory[i] = z[_][i]

# Arguments
- `pointer::UInt64`: the pointer.
- `register_index::Integer`: Y register index.
- `register_count::RegisterCount`: Either one, two or four
"""
@inline function stzi(pointer::UInt64; register_index::Integer=0, right_hand_half::Bool=false)  
    amx_stzi(amx_operands_memory_zi(pointer, right_hand_half, register_index))
end