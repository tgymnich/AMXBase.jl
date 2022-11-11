using BitOperations

@inline ptr_row_flags(ptr, row, flags) = ptr + ((row + flags * 64) << 56)

@inline ptr_row(ptr, row) = (row << 56) | ptr

@inline function amx_operands_memory_xy(pointer, register_index, double_width)
    op = UInt64(0)
    op = bset(op, 0:55, pointer)
    op = bset(op, 56:58, register_index)
    op = bset(op, 62, double_width)
    return op
end

@inline function amx_operands_memory_z(pointer, row_Z, double_width)
    op = UInt64(0)
    op = bset(op, 0:55, pointer)
    op = bset(op, 56:61, row_z)
    op = bset(op, 62, double_width)
    return op
end

@inline function amx_operands_memory_zi(pointer, right_hand_half, row_z)
    op = UInt64(0)
    op = bset(op, 0:55, pointer)
    op = bset(op, 56, right_hand_half)
    op = bset(op, 56:61, row_z)
    return op
end


@enum XY_Enable_Mode begin
    all_lanes = 0 # Enable all lanes (enable_x/y: 0), or odd lanes only (enable_x/y: 1), or even lanes only (enable_x/y: 2),
    lane_N = 1
    first_N_lanes = 2
    last_N_lanes = 3
end

"""
*offset_y:* offset into the y register pool in bytes
*offset_x:* offset into the y register pool in bytes

x*y+z
"""
@inline function amx_operands_floating_point_arithmetic(offset_y, offset_x, row_z, skip_z, skip_y, skip_x, 
                                                        enable_y, enable_mode_y, enable_x, enable_mode_x,
                                                        y_f16, x_f16, z_f32, vector_mode)
    op = UInt64(0)
    op = bset(op, 0:8, offset_y)
    op = bset(op, 10:18, offset_x)
    op = bset(op, 20:28, row_z)
    op = bset(op, 27, skip_z)
    op = bset(op, 28, skip_y)
    op = bset(op, 29, skip_x)
    op = bset(op, 32:36, enable_y)
    op = bset(op, 37:38, enable_mode_y)
    op = bset(op, 41:45, enable_x)
    op = bset(op, 46:47, enable_mode_x)
    op = bset(op, 60, y_f16)
    op = bset(op, 61, x_f16)
    op = bset(op, 62, z_f32)
    op = bset(op, 63, vector_mode)
    return op
end

"""
z[j][i] ±= f(x[i], y[j])
"""
@inline function amx_operands_floating_point_arithmetic_shuffle(offset_y, offset_x, row_z, skip_z, skip_y, skip_x, 
                                                                enable_y, enable_mode_y, enable_x, enable_mode_x,
                                                                y_f16, x_f16, z_f32, vector_mode)
end

"""
z+((x*y)>>s)
"""
@inline function amx_operands_integer_arithmetic(offset_y, offset_x, row_z, skip_z, skip_y, skip_x, 
                                                 enable_y, enable_mode_y, enable_x, enable_mode_x,
                                                 right_shift, y_i8, x_i8, z_i32, vector_mode)
    op = UInt64(0)
    op = bset(op, 0:8, offset_y)
    op = bset(op, 10:18, offset_x)
    op = bset(op, 20:28, row_z)
    op = bset(op, 27, skip_z)
    op = bset(op, 28, skip_y)
    op = bset(op, 29, skip_x)
    op = bset(op, 32:36, enable_y)
    op = bset(op, 37:38, enable_mode_y)
    op = bset(op, 41:45, enable_x)
    op = bset(op, 46:47, enable_mode_x)
    op = bset(op, 55:59, right_shift)
    op = bset(op, 60, y_i8)
    op = bset(op, 61, x_i8)
    op = bset(op, 62, z_i32)
    op = bset(op, 63, vector_mode)
    return op
end


@enum ALU_Modes fma = 0 fms = 1 relu = 4

@enum Lane_Width_Modes begin
 all_rows_interleaved_pairs = 3 
 one_row_from_each_four = 4
 one_row_from_each_eight = 7 
 one_row_from_each_two = 6 # using 6 as a placehoder for anything
end

@enum XY_Enable_Modes_Int begin
    first_n_lanes = 4
    last_n_lanes = 5
    no_langes = 6
end

"""
z[j][i] ±= f(x[i], y[j])
z[j][i]  = f(z[j][i])
"""
@inline function amx_operands_integer_arithmetic_shuffle(offset_y, offset_x, row_z, skip_z, skip_y, skip_x, 
                                                 enable_y, enable_mode_y, enable_x, enable_mode_x,
                                                 right_shift, y_i8, x_i8, z_i32, vector_mode)
end

@inline function amx_operands_extract(index_x, index_y)
    op = UInt64(0)
    op = bset(op, 16:18, index_x)
    op = bset(op, 20:22, index_y)
    op = bset(op, 26, 0)
    op = bset(op, 27, 1)
    return op
end


@inline function amx_operands_other(source_offset, source_y, dest_index_lo, dest_index_hi, 
                                    dest_y, dest_z, mode, table_from_y, table_index)
    @assert xor(dest_y,dest_z)

    op = UInt64(0)
    op = bset(op, 0:8, source_offset)
    op = bset(op, 10, source_y)
    op = bset(op, 20:22, dest_index_lo)
    op = bset(op, 23:25, dest_index_hi)
    op = bset(op, 25, dest_y)
    op = bset(op, 26, dest_z)
    op = bset(op, 53:56, mode)
    op = bset(op, 59, table_from_y)
    op = bset(op, 60:62, table_index)
    return op
end