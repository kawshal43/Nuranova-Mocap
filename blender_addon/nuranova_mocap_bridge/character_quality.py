"""Character-specific deformed-surface diagnostics and reversible limb corrections.

Triangle crossings are reported separately from containment estimates on capped regions.
Unresolved contacts remain in the report; smoothing is never a substitute for validation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json

import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree
import numpy as np

from .character_export import bound_meshes


def _region(canonical):
    for side in ("left", "right"):
        if canonical.startswith(side + "_"):
            suffix = canonical[len(side) + 1:]
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                if suffix.startswith(finger):
                    return side + "_" + finger
            if suffix in {"clavicle", "upper_arm"}:
                return side + "_upper_arm"
            if suffix in {"forearm", "hand", "upper_leg", "lower_leg"}:
                return side + "_" + suffix
            if suffix == "hand_end":
                return side + "_hand"
            if suffix in {"foot", "toe"}:
                return side + "_foot"
    return "head" if canonical in {"head", "head_end", "neck"} else "torso"


def rig_signature(armature):
    digest = hashlib.sha256()
    for bone in armature.data.bones:
        digest.update(bone.name.encode())
        digest.update(np.asarray(bone.matrix_local, np.float32).tobytes())
    return digest.hexdigest()


def mesh_signature(mesh):
    digest = hashlib.sha256()
    coordinates = np.empty(len(mesh.data.vertices) * 3, np.float32)
    mesh.data.vertices.foreach_get("co", coordinates)
    digest.update(coordinates.tobytes())
    digest.update(json.dumps([group.name for group in mesh.vertex_groups]).encode())
    digest.update(np.asarray([(v.index, g.group, g.weight) for v in mesh.data.vertices for g in v.groups], np.float64).tobytes())
    digest.update(json.dumps([list(poly.vertices) for poly in mesh.data.polygons]).encode())
    return digest.hexdigest()


def build_character_profile(armature, bone_mapping, margin_m=.003):
    if armature is None or armature.type != "ARMATURE":
        raise ValueError("Choose a mapped character armature.")
    meshes = bound_meshes(armature)
    if not meshes:
        raise ValueError("No bound character meshes were found for this armature.")
    missing = [name for name in ("hips", "left_upper_arm", "left_forearm", "left_hand", "right_upper_arm",
               "right_forearm", "right_hand", "left_upper_leg", "left_lower_leg", "left_foot",
               "right_upper_leg", "right_lower_leg", "right_foot") if bone_mapping.get(name) not in armature.data.bones]
    if missing:
        raise ValueError("Map the character's limb bones before analyzing contacts: " + ", ".join(missing))
    reverse = {target: _region(source) for source, target in bone_mapping.items()}
    # Unmapped helper/twist bones inherit their nearest mapped ancestor's anatomical region.
    for bone in armature.data.bones:
        ancestor = bone
        while ancestor and ancestor.name not in reverse:
            ancestor = ancestor.parent
        if ancestor:
            reverse.setdefault(bone.name, reverse[ancestor.name])
    profile = {"schema_version": 1, "armature": armature.name, "rig_signature": rig_signature(armature),
               "mapping": dict(bone_mapping), "margin_m": float(margin_m),
               "unit_scale": bpy.context.scene.unit_settings.scale_length,
               "meshes": {}, "warnings": [], "contact_pins": [], "floor_z_m": 0.}
    if min(armature.scale) <= 0:
        profile["warnings"].append("Negative object scale: check mirrored bone rolls and exported skinning.")
    constraints = [pb.name for pb in armature.pose.bones if any(not c.mute and c.influence > 0 for c in pb.constraints)]
    if constraints:
        profile["warnings"].append("Active rig constraints need a visual bake before direct deform-bone correction: " + ", ".join(constraints[:8]))
    for obj in meshes:
        labels, confidence = [], []
        groups = {g.index: reverse[g.name] for g in obj.vertex_groups if g.name in reverse}
        unweighted = 0
        for vertex in obj.data.vertices:
            weights = defaultdict(float)
            for weight in vertex.groups:
                if weight.group in groups:
                    weights[groups[weight.group]] += weight.weight
            if weights:
                label = max(weights, key=weights.get)
                labels.append(label)
                confidence.append(weights[label] / max(sum(weights.values()), 1e-8))
            else:
                labels.append("unmapped")
                confidence.append(0.)
                unweighted += 1
        obj.data.calc_loop_triangles()
        triangles = [list(triangle.vertices) for triangle in obj.data.loop_triangles]
        regions = defaultdict(list)
        for index, triangle in enumerate(triangles):
            label, count = Counter(labels[v] for v in triangle).most_common(1)[0]
            if count >= 2 and label != "unmapped":
                regions[label].append(index)
        profile["meshes"][obj.name] = {"vertices": len(labels), "triangles": triangles,
            "regions": dict(regions), "labels": labels, "confidence": confidence,
            "mesh_signature": mesh_signature(obj)}
        if unweighted:
            profile["warnings"].append(f"{obj.name}: {unweighted} vertices have no mapped deform weights.")
        for modifier in obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == armature and not modifier.use_deform_preserve_volume:
                profile["warnings"].append(f"{obj.name}: linear skinning; inspect axial twist before changing Preserve Volume.")
            if modifier.type == "ARMATURE" and modifier.object != armature and modifier.show_viewport:
                profile["warnings"].append(f"{obj.name}: additional armature {modifier.object.name if modifier.object else 'missing'}; export checks whether it is neutral or needs baking.")
        if obj.data.shape_keys:
            profile["warnings"].append(f"{obj.name}: animated shape keys are included in evaluated geometry.")
    return profile


def _closed_region(vertices, triangles):
    """Cap segmentation boundaries for containment diagnostics, retaining actual surface BVH."""
    boundary = {}
    for triangle in triangles:
        for a, b in zip(triangle, triangle[1:] + triangle[:1]):
            edge = tuple(sorted((a, b)))
            if edge in boundary:
                del boundary[edge]
            else:
                boundary[edge] = (a, b)
    points, faces = list(vertices), list(triangles)
    neighbors = defaultdict(set)
    for a, b in boundary.values():
        neighbors[a].add(b)
        neighbors[b].add(a)
    remaining = set(neighbors)
    while remaining:
        connected, pending = set(), [remaining.pop()]
        while pending:
            v = pending.pop()
            if v in connected:
                continue
            connected.add(v)
            pending.extend(neighbors[v] - connected)
        remaining -= connected
        center = np.mean([vertices[i] for i in connected], axis=0)
        center_index = len(points)
        points.append(tuple(center))
        for a, b in boundary.values():
            if a in connected:
                faces.append([b, a, center_index])
    return BVHTree.FromPolygons(points, faces, all_triangles=True)


def _inside(tree, point, max_distance):
    votes = 0
    for direction in (Vector((1., .137, .251)).normalized(), Vector((.213, 1., .173)).normalized()):
        origin = Vector(point) + direction * 1e-6
        distance, hits = max_distance, 0
        for _ in range(64):
            location, _, _, step = tree.ray_cast(origin, direction, distance)
            if location is None:
                break
            hits += 1
            distance -= step + 1e-5
            origin = location + direction * 1e-5
            if distance <= 0:
                break
        votes += hits % 2
    return votes == 2


def _neighbors(a, b):
    if a == b:
        return True
    for side in ("left", "right"):
        pairs = [("torso", side + "_upper_arm"), ("torso", side + "_upper_leg"),
                 (side + "_upper_arm", side + "_forearm"), (side + "_forearm", side + "_hand"),
                 (side + "_upper_leg", side + "_lower_leg"), (side + "_lower_leg", side + "_foot")]
        if any({a, b} == set(pair) for pair in pairs):
            return True
    return {a, b} == {"head", "torso"}


class CharacterGeometry:
    def __init__(self, armature, profile):
        if profile["rig_signature"] != rig_signature(armature):
            raise ValueError("Character rest bones changed; rebuild the collision profile.")
        self.armature, self.profile = armature, profile
        self.regions = {}
        self.neighborhoods = {}
        grouped = defaultdict(list)
        for name, mesh_profile in profile["meshes"].items():
            mesh = bpy.data.objects.get(name)
            if mesh is None or (mesh_profile.get("mesh_signature") and mesh_signature(mesh) != mesh_profile["mesh_signature"]):
                raise ValueError("Character skin weights or rest mesh changed; rebuild the collision profile.")
            adjacent = defaultdict(set)
            for triangle in mesh_profile["triangles"]:
                for vertex in triangle:
                    adjacent[vertex].update(triangle)
            self.neighborhoods[name] = {v: adjacent[v] | set().union(*(adjacent[n] for n in adjacent[v])) for v in adjacent}
            triangles = np.asarray(mesh_profile["triangles"], int)
            for region, indices in mesh_profile["regions"].items():
                ids, local = np.unique(triangles[indices], return_inverse=True)
                grouped[region].append((name, ids, local.reshape(-1, 3), np.asarray(mesh_profile["confidence"])[ids]))
        self.layouts = {}
        for region, pieces in grouped.items():
            count = 0
            triangles, identities, confidence = [], [], []
            for name, ids, local, weights in pieces:
                triangles.extend((local + count).tolist())
                identities.extend((name, int(i)) for i in ids)
                confidence.extend(weights)
                count += len(ids)
            self.layouts[region] = {"pieces":pieces, "triangles":triangles, "identities":identities,
                                    "confidence":np.asarray(confidence)}
        self._near_triangles = {}

    def update(self):
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        worlds = {}
        scale = self.profile["unit_scale"]
        for name, mesh_profile in self.profile["meshes"].items():
            obj = bpy.data.objects.get(name)
            if obj is None:
                raise ValueError("Character mesh was removed: " + name)
            evaluated = obj.evaluated_get(graph)
            mesh = evaluated.data
            if len(mesh.vertices) != mesh_profile["vertices"]:
                raise ValueError(f"{name}: a modifier changed topology; bake/profile that mesh separately.")
            coordinates = np.empty(len(mesh.vertices) * 3, np.float32)
            mesh.vertices.foreach_get("co", coordinates)
            matrix = np.asarray(evaluated.matrix_world)
            worlds[name] = (coordinates.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]) * scale
        self.regions = {}
        for region, layout in self.layouts.items():
            array = np.concatenate([worlds[name][ids] for name, ids, _, _ in layout["pieces"]])
            self.regions[region] = {**layout, "points":array, "low": array.min(axis=0), "high": array.max(axis=0),
                "surface": None, "closed": None}

    def _trees(self, region):
        data = self.regions[region]
        if data["surface"] is None:
            vertices = data["points"].tolist()
            data["surface"] = BVHTree.FromPolygons(vertices, data["triangles"], all_triangles=True)
        return data

    def analyze(self, frame=None, samples_per_region=40, pairs=None):
        self.update()
        margin = self.profile["margin_m"]
        contacts = []
        frame = bpy.context.scene.frame_current_final if frame is None else float(frame)
        for a, b in itertools.combinations(self.regions, 2):
            if pairs is not None and frozenset((a, b)) not in pairs:
                continue
            if _neighbors(a, b):
                continue
            aa, bb = self.regions[a], self.regions[b]
            if np.any(aa["low"] > bb["high"] + margin) or np.any(bb["low"] > aa["high"] + margin):
                continue
            aa, bb = self._trees(a), self._trees(b)
            crossings = 0
            for ai, bi in aa["surface"].overlap(bb["surface"]):
                a_ids = {aa["identities"][i] for i in aa["triangles"][ai]}
                b_ids = {bb["identities"][i] for i in bb["triangles"][bi]}
                key = (a, ai)
                if key not in self._near_triangles:
                    self._near_triangles[key] = {(mesh, n) for mesh, vertex in a_ids for n in self.neighborhoods[mesh][vertex]}
                if not self._near_triangles[key].intersection(b_ids):
                    crossings += 1
            deepest, direction, moving, penetrating_point = 0., None, None, None
            for source_name, target_name, source, target in ((a, b, aa, bb), (b, a, bb, aa)):
                candidates = np.flatnonzero((source["confidence"] >= .7) &
                    (source["points"] >= target["low"] - margin).all(axis=1) &
                    (source["points"] <= target["high"] + margin).all(axis=1))
                if len(candidates) > samples_per_region:
                    candidates = candidates[np.linspace(0, len(candidates) - 1, samples_per_region).astype(int)]
                extent = float(np.linalg.norm(target["high"] - target["low"]) + 1)
                for index in candidates:
                    point = source["points"][index]
                    location, normal, _, distance = target["surface"].find_nearest(Vector(point))
                    if location is None or distance < .0003 or distance > .2:
                        continue
                    if np.dot(point - np.asarray(location), np.asarray(normal)) >= -.0003:
                        continue
                    if target["closed"] is None:
                        target["closed"] = _closed_region(target["points"].tolist(), target["triangles"])
                    if _inside(target["closed"], point, extent) and distance > deepest:
                        deepest, direction, moving = float(distance), list(normal), source_name
                        penetrating_point = point.tolist()
            # Exact intersections detect thin crossings even if vertex sampling misses them.
            if crossings or deepest > .001:
                contacts.append({"a": a, "b": b, "depth_m": deepest,
                    "triangle_crossings": crossings, "moving_region": moving, "push_normal": direction,
                    "penetrating_point_m":penetrating_point,
                    "kind": "surface_crossing" if crossings else "capped_region_containment"})
        floor = float(self.profile.get("floor_z_m", 0.))
        for side in ("left", "right"):
            region = side + "_foot"
            if region in self.regions:
                penetration = floor - float(self.regions[region]["low"][2])
                if penetration > .001:
                    contacts.append({"a": region, "b": "floor", "depth_m": penetration,
                        "triangle_crossings": 0, "moving_region": region, "push_normal": [0., 0., 1.], "kind": "floor"})
        return {"frame": frame, "contacts": contacts, "max_depth_m": max((c["depth_m"] for c in contacts), default=0.),
                "triangle_crossings": sum(c["triangle_crossings"] for c in contacts)}


def duplicate_character(armature):
    """Copy deform data/actions/meshes. Original controls and actions remain recoverable."""
    collection = bpy.data.collections.new(armature.name + "_ContactReview")
    bpy.context.scene.collection.children.link(collection)
    duplicate = armature.copy()
    duplicate.data = armature.data.copy()
    duplicate.name = armature.name + "_ContactCorrected"
    duplicate["nuranova_original_rig"] = armature.name
    duplicate["nuranova_character_profile"] = duplicate.name + "_ContactProfile.json"
    collection.objects.link(duplicate)
    duplicate.parent = None
    duplicate.matrix_world = armature.matrix_world.copy()
    if duplicate.animation_data:
        if duplicate.animation_data.action:
            duplicate.animation_data.action = duplicate.animation_data.action.copy()
            duplicate.animation_data.action.name = duplicate.name + "_Action"
        duplicate.animation_data.use_nla = False
    mesh_map = {}
    for mesh in bound_meshes(armature):
        copied = mesh.copy()
        copied.data = mesh.data.copy()
        copied.name = mesh.name + "_ContactCorrected"
        collection.objects.link(copied)
        copied.parent = duplicate if mesh.parent == armature else mesh.parent
        copied.matrix_world = mesh.matrix_world.copy()
        for modifier in copied.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == armature:
                modifier.object = duplicate
        mesh_map[mesh.name] = copied.name
    return duplicate, mesh_map


def _world_head(armature, name):
    return armature.matrix_world @ armature.pose.bones[name].head


def _orient_segment(armature, bone_name, child_name, desired):
    bone = armature.pose.bones[bone_name]
    head, child = _world_head(armature, bone_name), _world_head(armature, child_name)
    current = child - head
    if current.length < 1e-6 or desired.length < 1e-6:
        return
    delta = current.rotation_difference(desired)
    world = armature.matrix_world @ bone.matrix
    location, rotation, scale = world.decompose()
    local_location, local_scale = bone.location.copy(), bone.scale.copy()
    bone.matrix = armature.matrix_world.inverted() @ Matrix.LocRotScale(location, delta @ rotation, scale)
    bone.location, bone.scale = local_location, local_scale
    bpy.context.view_layer.update()


def solve_character_limb(armature, mapping, side, limb, goal):
    names = [mapping.get(side + "_" + suffix) for suffix in
             (("upper_arm", "forearm", "hand") if limb == "arm" else ("upper_leg", "lower_leg", "foot"))]
    if any(name not in armature.pose.bones for name in names):
        return []
    root, mid, end = [_world_head(armature, name) for name in names]
    upper, lower = (mid - root).length, (end - mid).length
    axis = goal - root
    distance = max(abs(upper - lower) + 1e-5, min(axis.length, upper + lower - 1e-5))
    if axis.length < 1e-6:
        return []
    axis.normalize()
    pole = mid - root
    pole -= axis * pole.dot(axis)
    if pole.length < 1e-5:
        pole = axis.orthogonal()
    pole.normalize()
    along = (upper * upper + distance * distance - lower * lower) / (2 * distance)
    height = max(0., upper * upper - along * along) ** .5
    elbow = root + axis * along + pole * height
    end_world_rotation = (armature.matrix_world @ armature.pose.bones[names[2]].matrix).to_quaternion()
    _orient_segment(armature, names[0], names[1], elbow - root)
    _orient_segment(armature, names[1], names[2], root + axis * distance - _world_head(armature, names[1]))
    hand = armature.pose.bones[names[2]]
    world = armature.matrix_world @ hand.matrix
    location, _, scale = world.decompose()
    local_location, local_scale = hand.location.copy(), hand.scale.copy()
    hand.matrix = armature.matrix_world.inverted() @ Matrix.LocRotScale(location, end_world_rotation, scale)
    hand.location, hand.scale = local_location, local_scale
    bpy.context.view_layer.update()
    return names


def fix_character_frame(geometry, frame, iterations=3, max_shift_m=.10):
    armature, profile = geometry.armature, geometry.profile
    from .source_armature import _get_action_fcurves
    action = armature.animation_data.action if armature.animation_data else None
    if action and any("rotation_euler" in curve.data_path for curve in _get_action_fcurves(action,armature)):
        raise ValueError("Visually bake Euler-keyed animation before contact correction.")
    for bone in armature.pose.bones:
        if bone.rotation_mode != "QUATERNION":
            basis = bone.matrix_basis.copy()
            bone.rotation_mode = "QUATERNION"
            bone.matrix_basis = basis
    baseline = geometry.analyze(frame)
    before = {pb.name: pb.matrix_basis.copy() for pb in armature.pose.bones}
    before_scales = {pb.name: np.asarray(pb.scale).copy() for pb in armature.pose.bones}
    before_locations = {pb.name: np.asarray(pb.location).copy() for pb in armature.pose.bones}
    original_quaternions = {pb.name: pb.rotation_quaternion.copy() for pb in armature.pose.bones}
    original_anchors = {side + "_" + end: _world_head(armature, profile["mapping"][side + "_" + end])
                        for side in ("left", "right") for end in ("hand", "foot")}
    changed, unresolved = set(), []
    report = baseline
    anchors = profile.get("flat_stance_anchors", {}).get(str(int(frame)), {})
    for side, anchor in anchors.items():
        goal = Vector(anchor)
        current = original_anchors[side + "_foot"]
        if (goal - current).length * profile["unit_scale"] <= .10:
            changed.update(solve_character_limb(armature, profile["mapping"], side, "leg", goal))
    if changed:
        report = geometry.analyze(frame)
    for _ in range(iterations):
        did_change = False
        moved_endpoints = set()
        for contact in sorted(report["contacts"], key=lambda c: c["depth_m"], reverse=True):
            finger_regions = [r for r in (contact["a"], contact["b"])
                              if any(r.endswith("_" + f) for f in ("thumb", "index", "middle", "ring", "pinky"))]
            internal_hand = all(any(r.endswith("_" + suffix) for suffix in
                                ("hand", "thumb", "index", "middle", "ring", "pinky"))
                                for r in (contact["a"], contact["b"]))
            if finger_regions and internal_hand:
                finger_changed = _adjust_finger_contact(geometry, contact, frame)
                if finger_changed:
                    changed.update(finger_changed)
                    did_change = True
                    continue
            if contact["depth_m"] <= .001 or not contact["push_normal"]:
                continue
            moving = contact["moving_region"]
            preferred = next((r for r in (contact["a"], contact["b"]) if
                any(r.endswith("_" + suffix) for suffix in ("hand", "forearm", "thumb", "index", "middle", "ring", "pinky"))), None)
            direction = Vector(contact["push_normal"])
            if preferred and moving != preferred:
                moving = preferred
                direction.negate()
            side = "left" if moving.startswith("left_") else "right" if moving.startswith("right_") else None
            if side is None:
                other = contact["b"] if moving == contact["a"] else contact["a"]
                side = "left" if other.startswith("left_") else "right" if other.startswith("right_") else None
                if side:
                    moving = other
                    direction = -Vector(contact["push_normal"])
                else:
                    unresolved.append(contact)
                    continue
            leg = any(part in moving for part in ("leg", "foot"))
            end = side + ("_foot" if leg else "_hand")
            if end in moved_endpoints:
                continue
            pins = [p for p in profile.get("contact_pins", []) if p["start"] <= frame <= p["end"] and
                    p.get("kind") == "pin" and end in {p["a"], p["b"]}]
            if pins:
                unresolved.append(contact)
                continue
            current = _world_head(armature, profile["mapping"][end])
            touch = any(p["start"] <= frame <= p["end"] and p.get("kind") == "touch"
                        and end in {p["a"], p["b"]} for p in profile.get("contact_pins", []))
            shift = direction * min(.035, contact["depth_m"] + (.0003 if touch else profile["margin_m"]))
            goal = current + shift / profile["unit_scale"]
            if leg and side in anchors:
                goal.x, goal.y = anchors[side][0], anchors[side][1]
            displacement = goal - original_anchors[end]
            limit = max_shift_m / profile["unit_scale"]
            if displacement.length > limit:
                goal = original_anchors[end] + displacement.normalized() * limit
            changed.update(solve_character_limb(armature, profile["mapping"], side, "leg" if leg else "arm", goal))
            moved_endpoints.add(end)
            did_change = True
        if not did_change:
            break
        candidate = geometry.analyze(frame)
        # Reject a correction that makes the total penetration worse.
        old_score = sum(c["depth_m"] ** 2 for c in report["contacts"])
        new_score = sum(c["depth_m"] ** 2 for c in candidate["contacts"])
        if new_score > old_score + .000001:
            for name, matrix in before.items():
                armature.pose.bones[name].matrix_basis = matrix
            bpy.context.view_layer.update()
            changed.clear()
            report = baseline
            break
        report = candidate
    invalid_transform = any(not np.isfinite(np.asarray(bone.matrix_basis)).all() or
        not np.allclose(np.asarray(bone.scale),before_scales[bone.name],atol=1e-5,rtol=1e-5) or
        not np.allclose(np.asarray(bone.location),before_locations[bone.name],atol=1e-5,rtol=1e-5)
        for bone in armature.pose.bones)
    invalid_reach = any((_world_head(armature,profile["mapping"][name])-anchor).length * profile["unit_scale"] > max_shift_m+.002
                        for name,anchor in original_anchors.items())
    if invalid_transform or invalid_reach:
        for name,matrix in before.items():
            armature.pose.bones[name].matrix_basis=matrix
        bpy.context.view_layer.update()
        return {"frame":float(frame),"before":baseline,"after":baseline,"changed_bones":[],
                "unresolved":baseline["contacts"],"status":"rejected unsafe correction",
                "reason":"A correction changed local scale/translation or exceeded the endpoint displacement limit."}
    for name in changed:
        bone = armature.pose.bones[name]
        bone.rotation_mode = "QUATERNION"
        if bone.rotation_quaternion.dot(original_quaternions[name]) < 0:
            bone.rotation_quaternion.negate()
        bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group="Contact correction")
    return {"frame": float(frame), "before": baseline, "after": report, "changed_bones": sorted(changed),
            "unresolved": unresolved, "status": "review required" if report["contacts"] else "clear"}


def _adjust_finger_contact(geometry, contact, frame):
    armature = geometry.armature
    pair = {contact["a"], contact["b"]}
    def cost(contacts):
        return sum(c["depth_m"] ** 2 + c["triangle_crossings"] * 1e-8
                   for c in contacts if {c["a"], c["b"]} == pair)
    baseline_score = cost(geometry.analyze(frame, samples_per_region=20, pairs={frozenset(pair)})["contacts"])
    best = None
    for region in (contact["a"], contact["b"]):
        if not any(region.endswith("_" + f) for f in ("thumb", "index", "middle", "ring", "pinky")):
            continue
        analytic = contact.get("penetrating_point_m") is not None and contact["moving_region"] == region
        if contact.get("penetrating_point_m") is not None and not analytic:
            continue
        for segment in ("01", "02", "03"):
            if not analytic and segment != "01":
                continue
            name = geometry.profile["mapping"].get(region + "_" + segment)
            if name not in armature.pose.bones:
                continue
            bone = armature.pose.bones[name]
            original = bone.rotation_quaternion.copy()
            for axis in ((1, 0, 0), (0, 0, 1)) if segment == "01" else ((1, 0, 0),):
                angles = [-np.radians(3.), np.radians(3.)]
                if analytic:
                    world = armature.matrix_world @ bone.matrix
                    world_axis = (world.to_3x3() @ Vector(axis)).normalized()
                    lever = Vector(contact["penetrating_point_m"]) - world.translation * geometry.profile["unit_scale"]
                    jacobian = world_axis.cross(lever)
                    desired = Vector(contact["push_normal"]) * min(.02, contact["depth_m"] + .001)
                    angle = jacobian.dot(desired) / max(jacobian.length_squared, 1e-8)
                    angles = [float(np.clip(angle, -np.radians(3.), np.radians(3.)))]
                for angle in angles:
                    candidate = original @ Quaternion(Vector(axis), angle)
                    bone.rotation_quaternion = candidate
                    score = cost(geometry.analyze(frame, samples_per_region=20, pairs={frozenset(pair)})["contacts"])
                    if score < baseline_score * .98 and (best is None or score < best[0]):
                        best = score, name, candidate.copy()
            bone.rotation_quaternion = original
    if best:
        armature.pose.bones[best[1]].rotation_quaternion = best[2]
        bpy.context.view_layer.update()
        return [best[1]]
    return []


def regularize_contact_corrections(original, working, start, end, fps, max_rate_deg_s=180.):
    """Smooth only the added correction, preserving the performer's base motion.

    This runs before final mesh validation: bounded temporal changes can leave an
    infeasible contact, which must remain in the resulting geometry report.
    """
    from .source_armature import _get_action_fcurves
    if not working.animation_data or not working.animation_data.action:
        yield {"stage":"contact correction continuity", "regularized_bones":[], "max_correction_step_deg":0.}
        return
    names = [bone.name for bone in working.pose.bones]
    source, corrected = [], []
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        source.append([bone.matrix_basis.to_quaternion() for bone in original.pose.bones])
        corrected.append([working.pose.bones[name].matrix_basis.to_quaternion() for name in names])
        if frame % 50 == 0:
            yield {"frame":frame, "stage":"sampling contact correction continuity"}
    if len(source) < 2:
        return
    if [bone.name for bone in original.pose.bones] != names:
        raise ValueError("The correction copy must retain the original bone order.")
    curves = _get_action_fcurves(working.animation_data.action, working)
    alpha = float(1 - np.exp(-2 * np.pi * 6 / fps))
    limit = float(np.radians(max_rate_deg_s / fps))
    changed, largest_step = [], 0.
    action_range = original.animation_data.action.frame_range if original.animation_data and original.animation_data.action else (start,end)
    for i, name in enumerate(names):
        delta = [(a[i].conjugated() @ b[i]).normalized() for a,b in zip(source,corrected)]
        if max(2 * np.arccos(np.clip(abs(q.w),0,1)) for q in delta) < 1e-5:
            continue
        for t in range(1,len(delta)):
            if delta[t-1].dot(delta[t]) < 0:
                delta[t].negate()
            delta[t] = delta[t-1].slerp(delta[t],alpha)
        for t in range(len(delta)-2,-1,-1):
            delta[t] = delta[t+1].slerp(delta[t],alpha)
        if start > action_range[0]:
            delta[0] = Quaternion()
        if end < action_range[1]:
            delta[-1] = Quaternion()
        vectors = []
        for q in delta:
            if q.w < 0:
                q.negate()
            vectors.append(tuple(q.to_exponential_map()))
        vectors = np.asarray(vectors)
        maximum = float(np.linalg.norm(np.diff(vectors,axis=0),axis=1).max())
        # A uniform contraction in the rotation tangent space preserves zero
        # boundary corrections and bounds every added angular step.
        if maximum > limit:
            vectors *= limit / maximum
        delta = [Quaternion(Vector(value)) for value in vectors]
        largest_step = max(largest_step,max(2*np.arccos(np.clip(abs(a.dot(b)),0,1)) for a,b in zip(delta,delta[1:])))
        rotations = [source[t][i] @ delta[t] for t in range(len(delta))]
        for t in range(1,len(rotations)):
            if rotations[t-1].dot(rotations[t]) < 0:
                rotations[t].negate()
        bone = working.pose.bones[name]
        bone.rotation_mode = "QUATERNION"
        path = bone.path_from_id("rotation_quaternion")
        for axis in range(4):
            curve = curves.find(path,index=axis)
            if curve is None:
                curve = curves.new(path,index=axis)
            for frame,rotation in zip(range(start,end+1),rotations):
                key = curve.keyframe_points.insert(frame,rotation[axis],options={"FAST"})
                key.interpolation = "LINEAR"
            curve.update()
        changed.append(name)
    bpy.context.scene.frame_set(start)
    yield {"stage":"contact correction continuity", "regularized_bones":changed,
           "max_correction_step_deg":float(np.degrees(largest_step)), "max_correction_rate_deg_s":max_rate_deg_s}


def visual_bake_copy(source, start, end):
    """Generator: bake evaluated controls/constraints into a separate deform rig and meshes."""
    from .source_armature import write_fcurve_batch
    duplicate, mesh_map = duplicate_character(source)
    for pb in duplicate.pose.bones:
        for constraint in list(pb.constraints):
            pb.constraints.remove(constraint)
    for constraint in list(duplicate.constraints):
        duplicate.constraints.remove(constraint)
    if duplicate.animation_data:
        for driver in duplicate.animation_data.drivers:
            driver.mute = True
    times = list(range(start, end + 1))
    samples = {pb.name: [] for pb in source.pose.bones}
    object_samples = []
    for frame in times:
        bpy.context.scene.frame_set(frame)
        evaluated = source.evaluated_get(bpy.context.evaluated_depsgraph_get())
        object_samples.append(evaluated.matrix_world.copy())
        baked_poses = {}
        for bone in sorted(source.data.bones, key=lambda item: len(item.parent_recursive)):
            pose = evaluated.pose.bones[bone.name]
            kwargs = {"parent_matrix": baked_poses[bone.parent.name],
                      "parent_matrix_local": bone.parent.matrix_local} if bone.parent else {}
            basis = bone.convert_local_to_pose(pose.matrix, bone.matrix_local, invert=True, **kwargs)
            components = basis.decompose()
            samples[bone.name].append(components)
            baked_poses[bone.name] = bone.convert_local_to_pose(Matrix.LocRotScale(*components),
                bone.matrix_local, **kwargs)
        yield {"frame": frame, "working": duplicate}
    action = bpy.data.actions.new(duplicate.name + "_VisualBake")
    duplicate.animation_data_create()
    duplicate.animation_data.action = action
    for name, transforms in samples.items():
        duplicate.pose.bones[name].rotation_mode = "QUATERNION"
        rotations = [q.copy() for _, q, _ in transforms]
        for i in range(1, len(rotations)):
            if rotations[i - 1].dot(rotations[i]) < 0:
                rotations[i].negate()
        for data_path, values in (("location", [loc for loc, _, _ in transforms]),
                                  ("rotation_quaternion", rotations), ("scale", [scale for _, _, scale in transforms])):
            for axis in range(len(values[0])):
                write_fcurve_batch(action, duplicate, duplicate.pose.bones[name].path_from_id(data_path),
                                   axis, times, [v[axis] for v in values])
    object_rotations = [matrix.to_quaternion() for matrix in object_samples]
    for i in range(1, len(object_rotations)):
        if object_rotations[i - 1].dot(object_rotations[i]) < 0:
            object_rotations[i].negate()
    duplicate.rotation_mode = "QUATERNION"
    for data_path, values in (("location", [m.to_translation() for m in object_samples]),
                             ("rotation_quaternion", object_rotations), ("scale", [m.to_scale() for m in object_samples])):
        for axis in range(len(values[0])):
            write_fcurve_batch(action, duplicate, data_path, axis, times, [v[axis] for v in values])
    yield {"complete": True, "working": duplicate, "mesh_map": mesh_map}


def save_profile(armature, profile):
    name = armature.get("nuranova_character_profile", armature.name + "_ContactProfile.json")
    text = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    text.clear()
    text.write(json.dumps(profile))
    armature["nuranova_character_profile"] = text.name
    return text.name


def load_profile(armature):
    name = armature.get("nuranova_character_profile")
    if not name or name not in bpy.data.texts:
        raise ValueError("Analyze this character to build its mesh collision profile first.")
    return json.loads(bpy.data.texts[name].as_string())
