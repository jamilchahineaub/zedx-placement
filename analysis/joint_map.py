# analysis/joint_map.py
#
# ZED BODY_18 keypoint <-> Isaac character joint-name mapping. Plain python3.
#
# The ZED side comes from sl.BODY_18_PARTS (18 keypoints, indices 0-17). The
# Isaac side is the `male_adult_police_04` rig in the reference scene (101
# joints; names taken from a real ground_truth CSV, see
# results/layouts/ground_truth_test_002.csv). Each BODY_18 keypoint is a JOINT
# CENTER, and the GT logger records each Isaac bone's origin in world space, so
# the natural pairing is keypoint -> the bone whose origin sits at that joint
# (e.g. the shoulder keypoint is the origin of the Upperarm bone).
#
# Keypoints with no anatomical counterpart in the rig map to None and are
# excluded from MPJPE/PCK.

# ZED BODY_18 name -> Isaac joint name (or None = unmapped).
# ZED names verified against real zed_single output (sl.BODY_18_PARTS uses the
# spelled-out LEFT_/RIGHT_ prefixes, e.g. "RIGHT_SHOULDER").
ZED18_TO_ISAAC = {
    "NOSE":           None,           # rig has no nose joint
    "NECK":           "NeckTwist01",  # base of the neck
    "RIGHT_SHOULDER": "R_Upperarm",   # shoulder joint = upper-arm bone origin
    "RIGHT_ELBOW":    "R_Forearm",
    "RIGHT_WRIST":    "R_Hand",
    "LEFT_SHOULDER":  "L_Upperarm",
    "LEFT_ELBOW":     "L_Forearm",
    "LEFT_WRIST":     "L_Hand",
    "RIGHT_HIP":      "R_Thigh",      # hip joint = thigh bone origin
    "RIGHT_KNEE":     "R_Calf",
    "RIGHT_ANKLE":    "R_Foot",
    "LEFT_HIP":       "L_Thigh",
    "LEFT_KNEE":      "L_Calf",
    "LEFT_ANKLE":     "L_Foot",
    "RIGHT_EYE":      "R_Eye",
    "LEFT_EYE":       "L_Eye",
    "RIGHT_EAR":      None,           # no ear joints in the rig
    "LEFT_EAR":       None,
}


def mapped_pairs():
    """[(zed_name, isaac_name)] for every mapped keypoint (None entries dropped)."""
    return [(z, i) for z, i in ZED18_TO_ISAAC.items() if i is not None]


def isaac_names():
    """Isaac joint names used by the mapping (for GT filtering)."""
    return [i for i in ZED18_TO_ISAAC.values() if i is not None]
