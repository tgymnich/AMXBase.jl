using BitOperations
using CEnum

### Floating-point matrix arithmetic

# x*y+z
@cenum ALUOperation::UInt8 begin
    xyz = 0
    xy = 1
    xz = 2
    x = 3
    yz = 4
    y = 5
    z = 6
    zero = 7
end

export LaneMode
struct LaneMode
    value::UInt8
    mode::UInt8
end

export all_lanes, odd_lanes, even_lanes, lane, first_lanes, last_lanes

all_lanes() = LaneMode(UInt8(0), UInt8(0))
odd_lanes() = LaneMode(UInt8(0), UInt8(1))
even_lanes() = LaneMode(UInt8(0), UInt8(2))
lane(n::UInt8) = LaneMode(UInt8(1), UInt8(n))
first_lanes(n::UInt8) = LaneMode(UInt8(2), UInt8(n))
last_lanes(n::UInt8) = LaneMode(UInt8(3), UInt8(n))


export fma64, fma32, fma16

"""
z[j][i] += x[i] * y[j]

# Arguments
- `offset_x::Integer`:
- `offset_y::Integer`: 
- `offset_z::Integer`: 
- `row_z::Integer`: 
- `op::ALUOperation`: 
- `lane_x::LaneMode`: 
- `lane_y::LaneMode`: 
- `x_f16::Bool`: 
- `y_f16::Bool`: 
- `vector_mode::Bool`: 
"""
@inline function fma64(;offset_x::Integer=0, offset_y::Integer=0, row_z::Integer=0, op::ALUOperation=xyz, lane_x::LaneMode=LaneMode(0,0), lane_y::LaneMode=LaneMode(0,0), x_f16::Bool=false, y_f16::Bool=false, vector_mode=false)  
    amx_fma64(amx_operands_floating_point_arithmetic(offset_y, offset_x, row_z, bget(UInt8(op), 0), bget(UInt8(op), 1), bget(UInt8(op), 2), lane_y.mode, lane_y.value, lane_x.mode, lane_x.value, y_f16, x_f16, false, vector_mode))
end


"""
z[j][i] += x[i] * y[j]

# Arguments
- `offset_x::Integer`:
- `offset_y::Integer`: 
- `offset_z::Integer`: 
- `row_z::Integer`: 
- `op::ALUOperation`: 
- `lane_x::LaneMode`: 
- `lane_y::LaneMode`: 
- `x_f16::Bool`: 
- `y_f16::Bool`: 
- `vector_mode::Bool`: 
"""
@inline function fma32(;offset_x::Integer=0, offset_y::Integer=0, row_z::Integer=0, op::ALUOperation=xyz, lane_x::LaneMode=LaneMode(0,0), lane_y::LaneMode=LaneMode(0,0), x_f16::Bool=false, y_f16::Bool=false, vector_mode=false)  
    amx_fma32(amx_operands_floating_point_arithmetic(offset_y, offset_x, row_z, bget(UInt8(op), 0), bget(UInt8(op), 1), bget(UInt8(op), 2), lane_y.mode, lane_y.value, lane_x.mode, lane_x.value, y_f16, x_f16, false, vector_mode))
end



"""
z[j][i] += x[i] * y[j]

# Arguments
- `offset_x::Integer`:
- `offset_y::Integer`: 
- `offset_z::Integer`: 
- `row_z::Unsigned`: 
- `op::ALUOperation`: 
- `lane_x::LaneMode`: 
- `lane_y::LaneMode`: 
- `x_f16::Bool`: 
- `y_f16::Bool`: 
- `z_f32::Bool`: 
- `vector_mode::Bool`: 
"""
@inline function fma16(;offset_x::Integer=0, offset_y::Integer=0, row_z::Integer=0, op::ALUOperation=xyz, lane_x::LaneMode=LaneMode(0,0), lane_y::LaneMode=LaneMode(0,0), x_f16::Bool=false, y_f16::Bool=false, z_f32::Bool=false, vector_mode=false)  
    amx_fma16(amx_operands_floating_point_arithmetic(offset_y, offset_x, row_z, bget(UInt8(op), 0), bget(UInt8(op), 1), bget(UInt8(op), 2), lane_y.mode, lane_y.value, lane_x.mode, lane_x.value, y_f16, x_f16, z_f32, vector_mode))
end