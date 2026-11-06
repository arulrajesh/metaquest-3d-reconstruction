"""CSV column headers written by QuestRealityCapture (MRUK backend and depth exporter).

Kept identical to the headset app so laptop stream recordings, synthetic
captures, and headset recordings are interchangeable.
"""

COLOR_METADATA_HEADER = (
    "frame_index", "file_name", "unix_time_ms", "timestamp_us_realtime", "width", "height", "format",
    "camera_position", "pose_pos_x", "pose_pos_y", "pose_pos_z", "pose_rot_x", "pose_rot_y", "pose_rot_z",
    "pose_rot_w", "pose_ok", "is_playing", "is_updated_this_frame", "current_resolution_width",
    "current_resolution_height", "pixel_count", "colors_array_length", "rgba_byte_count", "get_texture_ok",
    "get_colors_ok", "error",
)

STEREO_PAIR_HEADER = (
    "pair_index", "left_file_name", "right_file_name", "left_timestamp_us_realtime",
    "right_timestamp_us_realtime", "time_delta_us", "left_frame_index", "right_frame_index",
)

DEPTH_DESCRIPTOR_HEADER = (
    "timestamp_ms", "ovr_timestamp",
    "create_pose_location_x", "create_pose_location_y", "create_pose_location_z",
    "create_pose_rotation_x", "create_pose_rotation_y", "create_pose_rotation_z", "create_pose_rotation_w",
    "fov_left_angle_tangent", "fov_right_angle_tangent", "fov_top_angle_tangent", "fov_down_angle_tangent",
    "near_z", "far_z", "width", "height",
)
