# isaac/gt_logger.py
#
# Ground-truth joint logger. RUNS UNDER ISAAC PYTHON ONLY.
#
# Discovers the character's UsdSkel.Root / Skeleton, then each physics step logs
# every joint's WORLD position (Isaac Z-up, metres) with sim + wall-clock time.
#
# UsdSkel API confirmed against the Isaac 5.1 USD v0.24 build:
#   cache = UsdSkel.Cache()
#   cache.Populate(UsdSkel.Root(prim), Usd.PrimDefaultPredicate)
#   skel_query = cache.GetSkelQuery(UsdSkel.Skeleton(skel_prim))
#   joint_order = skel_query.GetJointOrder()                     # VtTokenArray
#   xf_cache = UsdGeom.XformCache(Usd.TimeCode(time))            # time enters HERE
#   world = skel_query.ComputeJointWorldTransforms(xf_cache)     # VtArray<Matrix4d>
# (Signature: ComputeJointWorldTransforms(UsdGeomXformCache*, bool atRest=False).)

import csv
import os


class GTLogger:
    def __init__(self, stage):
        from pxr import UsdSkel, Usd

        self.stage = stage
        self.rows = []            # (sim_time, wall_clock, joint_name, x, y, z)
        self._first_frame = True
        self._warned_pelvis = False

        # --- discover SkelRoot + Skeleton by traversing the stage ---
        self.cache = UsdSkel.Cache()
        skel_prim = None
        root_prim = None

        for prim in stage.Traverse():
            if prim.IsA(UsdSkel.Root):
                root_prim = prim
                self.cache.Populate(UsdSkel.Root(prim), Usd.PrimDefaultPredicate)
                break

        if root_prim is None:
            # Some characters author the Skeleton without a SkelRoot wrapper.
            for prim in stage.Traverse():
                if prim.IsA(UsdSkel.Skeleton):
                    skel_prim = prim
                    break
            if skel_prim is None:
                print("RUN_FAILED no UsdSkel.Root or UsdSkel.Skeleton found in stage")
                self.skel_query = None
                self.joint_names = []
                return
        else:
            # find the Skeleton under the root
            for prim in Usd.PrimRange(root_prim):
                if prim.IsA(UsdSkel.Skeleton):
                    skel_prim = prim
                    break

        if skel_prim is None:
            print("RUN_FAILED SkelRoot found but no Skeleton child")
            self.skel_query = None
            self.joint_names = []
            return

        self.skel_query = self.cache.GetSkelQuery(UsdSkel.Skeleton(skel_prim))
        # Joint order is a list of joint path tokens like "Hips/Spine/.../Head".
        # Use the leaf name as the joint name; keep full path available if needed.
        order = self.skel_query.GetJointOrder()
        self.joint_paths = [str(j) for j in order]
        self.joint_names = [p.split("/")[-1] for p in self.joint_paths]
        print(f"gt_logger: skeleton {skel_prim.GetPath()} with {len(self.joint_names)} joints")

    def _pelvis_index(self):
        """Index of the pelvis/hips joint, or None."""
        for i, n in enumerate(self.joint_names):
            ln = n.lower()
            if "pelvis" in ln or ln == "hips" or "hip" == ln:
                return i
        return None

    def log_frame(self, sim_time, wall_clock):
        """Convenience: treat sim_time (seconds) as the timecode too (only valid
        when TimeCodesPerSecond == 1). Prefer log_frame_at from run_episode."""
        self.log_frame_at(sim_time, wall_clock, sim_time)

    def log_frame_at(self, sim_seconds, wall_clock, usd_timecode):
        """
        sim_seconds  : episode time in seconds (stored in CSV)
        wall_clock   : time.time() (stored in CSV)
        usd_timecode : USD timecode (= sim_seconds * TimeCodesPerSecond) used to
                       sample the animated joint transforms.
        """
        if self.skel_query is None:
            return
        from pxr import UsdGeom, Usd

        xf_cache = UsdGeom.XformCache(Usd.TimeCode(usd_timecode))
        world_xforms = self.skel_query.ComputeJointWorldTransforms(xf_cache)
        if not world_xforms:
            return
        sim_time = sim_seconds

        for name, M in zip(self.joint_names, world_xforms):
            t = M.ExtractTranslation()
            self.rows.append((sim_time, wall_clock, name, float(t[0]), float(t[1]), float(t[2])))

        # First-frame sanity check on pelvis height (warn, do not crash).
        if self._first_frame:
            self._first_frame = False
            pi = self._pelvis_index()
            if pi is not None:
                pz = float(world_xforms[pi].ExtractTranslation()[2])
                if not (0.5 <= pz <= 2.0):
                    print(f"WARNING pelvis z={pz:.3f}m outside [0.5, 2.0] — "
                          f"character may be mis-scaled or not standing")
            else:
                print("WARNING no pelvis/hips joint found for first-frame height check")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sim_time", "wall_clock", "joint_name", "x", "y", "z"])
            w.writerows(self.rows)
        print(f"gt_logger: wrote {len(self.rows)} rows ({len(self.joint_names)} joints) -> {path}")
        return path
