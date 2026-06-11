# analysis/tests/test_joint_map.py

import joint_map

# Real joint names from results/layouts/ground_truth_test_002.csv
# (male_adult_police_04 rig, 101 joints) — the subset our mapping may target.
GT_JOINT_NAMES = {
    "Hip", "Pelvis", "Waist", "Spine01", "Spine02", "NeckTwist01", "NeckTwist02",
    "Head", "L_Eye", "R_Eye",
    "L_Clavicle", "L_Upperarm", "L_Forearm", "L_Hand",
    "R_Clavicle", "R_Upperarm", "R_Forearm", "R_Hand",
    "L_Thigh", "L_Calf", "L_Foot", "L_ToeBase",
    "R_Thigh", "R_Calf", "R_Foot", "R_ToeBase",
}

# Verified against real zed_single CSV output (sl.BODY_18_PARTS .name values).
ZED18_NAMES = {
    "NOSE", "NECK",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE",
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
    "RIGHT_EYE", "LEFT_EYE", "RIGHT_EAR", "LEFT_EAR",
}


def test_covers_all_body18_keys():
    assert set(joint_map.ZED18_TO_ISAAC) == ZED18_NAMES


def test_mapped_targets_exist_in_rig():
    for zed, isaac in joint_map.mapped_pairs():
        assert isaac in GT_JOINT_NAMES, f"{zed} -> {isaac} not a rig joint"


def test_left_right_consistency():
    """Every LEFT_* zed keypoint maps to an L_* rig joint, same for RIGHT_*."""
    for zed, isaac in joint_map.mapped_pairs():
        if zed.startswith("LEFT_"):
            assert isaac.startswith("L_"), f"{zed} -> {isaac}"
        if zed.startswith("RIGHT_"):
            assert isaac.startswith("R_"), f"{zed} -> {isaac}"


def test_mapped_count():
    # 18 keypoints minus NOSE and both EARs = 15 usable pairs
    assert len(joint_map.mapped_pairs()) == 15
    assert joint_map.ZED18_TO_ISAAC["NOSE"] is None
    assert joint_map.ZED18_TO_ISAAC["LEFT_EAR"] is None
    assert joint_map.ZED18_TO_ISAAC["RIGHT_EAR"] is None
