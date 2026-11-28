"""Explicit character/action exports with isolated scope and round-trip pose checks."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def bound_meshes(armature):
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and
            any(mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers)]


@contextmanager
def _fbx_motion_bind_pose(enabled):
    """Keep the rest skeleton in an FBX with no skinned mesh.

    Blender normally emits BindPose only for skinned meshes. Without it, a
    fresh importer can use the first animated pose as the skeleton's rest pose.
    Use the bundled exporter's own pose writer and restore its hooks afterward.
    """
    if not enabled:
        yield
        return
    from io_scene_fbx import export_fbx_bin as fbx
    previous_prepare = fbx.fbx_data_from_scene
    previous_armature = fbx.fbx_data_armature_elements

    def prepare(scene, graph, settings):
        data = previous_prepare(scene, graph, settings)
        old = data.templates.get(b"BindPose")
        data.templates[b"BindPose"] = fbx.fbx_template_def_pose(scene, settings, nbr_users=1)
        return data._replace(templates_users=data.templates_users + 1 - (old.nbr_users if old else 0))

    def write_armature(root, armature, data):
        previous_armature(root, armature, data)
        bones = tuple(bone for bone in armature.bones if bone in data.objects)
        fbx.fbx_data_bindpose_element(root, armature, armature.bdata.data, data,
                                     arm_obj=armature, bones=bones)

    try:
        fbx.fbx_data_from_scene, fbx.fbx_data_armature_elements = prepare, write_armature
        yield
    finally:
        fbx.fbx_data_from_scene, fbx.fbx_data_armature_elements = previous_prepare, previous_armature


def _neutral_auxiliary_rig(armature):
    animation = armature.animation_data
    if animation and (animation.action or animation.drivers or any(not t.mute for t in animation.nla_tracks)):
        return False
    if any(not c.mute and c.influence > 0 for c in armature.constraints):
        return False
    for pose in armature.pose.bones:
        if any(not c.mute and c.influence > 0 for c in pose.constraints):
            return False
        if not np.allclose(np.asarray(pose.matrix), np.asarray(pose.bone.matrix_local), atol=1e-6):
            return False
    return True


@contextmanager
def export_scope(armature, meshes, start, end, fps=None, static=False, include_armature=True):
    context, scene = bpy.context, bpy.context.scene
    active, selected = context.view_layer.objects.active, list(context.selected_objects)
    frame, subframe = scene.frame_current, scene.frame_subframe
    frame_range, rate = (scene.frame_start, scene.frame_end), (scene.render.fps, scene.render.fps_base)
    mode = context.object.mode if context.object else "OBJECT"
    visibility, animation, auxiliary_modifiers, shape_values = [], [], [], []
    old_layer = context.view_layer
    visibility = [(obj,obj.hide_get(view_layer=old_layer),obj.hide_viewport,obj.hide_render)
                  for obj in ([armature] if include_armature else [])+list(meshes)]
    scope_collection = scope_layer = None
    try:
        if mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        scope_collection = bpy.data.collections.new("Nuranova_Export_Scope")
        scene.collection.children.link(scope_collection)
        for obj in [armature,*meshes]:
            scope_collection.objects.link(obj)
            if obj.type=="MESH":
                for modifier in obj.modifiers:
                    if modifier.type=="ARMATURE" and modifier.object and modifier.object.name not in scope_collection.objects:
                        scope_collection.objects.link(modifier.object)
        scope_layer = scene.view_layers.new("Nuranova_Export")
        for child in scope_layer.layer_collection.children:
            child.exclude = child.collection != scope_collection
        if context.window:
            context.window.view_layer = scope_layer
        for obj in context.view_layer.objects:
            obj.select_set(False)
        for obj in ([armature] if include_armature else []) + list(meshes):
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_set(False)
            obj.select_set(True)
            if obj.type == "MESH" and obj.data.shape_keys:
                shape_values.extend((key,float(key.value)) for key in obj.data.shape_keys.key_blocks)
            if obj.type == "MESH":
                for modifier in obj.modifiers:
                    if modifier.type == "ARMATURE" and modifier.object and modifier.object != armature:
                        auxiliary_modifiers.append((modifier, modifier.object))
                        modifier.object = None
            for owner in [obj, obj.data.shape_keys if obj.type == "MESH" else None]:
                data = owner.animation_data if owner is not None else None
                if data is not None:
                    animation.append((data, data.use_nla, [(track, track.mute) for track in data.nla_tracks]))
                    data.use_nla = False
                    for track in data.nla_tracks:
                        track.mute = True
        context.view_layer.objects.active = armature
        scene.frame_start, scene.frame_end = int(start), int(end)
        if fps is not None:
            scene.render.fps = max(1, round(float(fps)))
            scene.render.fps_base = scene.render.fps / float(fps)
        if static:
            scene.frame_set(start)
        yield auxiliary_modifiers
    finally:
        for modifier, other_rig in auxiliary_modifiers:
            modifier.object = other_rig
        for data, use_nla, tracks in animation:
            data.use_nla = use_nla
            for track, mute in tracks:
                track.mute = mute
        for obj, hidden, viewport, render in visibility:
            obj.hide_viewport, obj.hide_render = viewport, render
            obj.hide_set(hidden,view_layer=old_layer)
        if context.window:
            context.window.view_layer = old_layer
        if scope_layer:
            scene.view_layers.remove(scope_layer)
        if scope_collection:
            bpy.data.collections.remove(scope_collection)
        for obj in context.view_layer.objects:
            obj.select_set(False)
        for obj in selected:
            obj.select_set(True)
        context.view_layer.objects.active = active
        scene.frame_start, scene.frame_end = frame_range
        scene.render.fps, scene.render.fps_base = rate
        scene.frame_set(frame, subframe=subframe)
        for key,value in shape_values:
            key.value=value
        if active is not None and mode != "OBJECT":
            bpy.ops.object.mode_set(mode=mode)


def _deformation_state(mesh, rigs):
    graph = bpy.context.evaluated_depsgraph_get()
    mesh_world = mesh.evaluated_get(graph).matrix_world
    values = []
    for rig in rigs:
        evaluated = rig.evaluated_get(graph)
        values.append(np.asarray(mesh_world.inverted_safe() @ evaluated.matrix_world).ravel())
        values.extend(np.asarray(bone.matrix).ravel() for bone in evaluated.pose.bones)
    return np.concatenate(values)


def _sample_pose(armature, meshes, frames, static_prefixes=()):
    positions, morphs, surfaces, deformations = [], [], [], []
    names = [bone.name for bone in armature.data.bones]
    shape_names = [(mesh.name, key.name) for mesh in meshes if mesh.data.shape_keys
                   for key in mesh.data.shape_keys.key_blocks if key != mesh.data.shape_keys.reference_key]
    sample_indices = {}
    for mesh in meshes:
        indices = set(np.linspace(0, max(0, len(mesh.data.vertices) - 1), min(96, len(mesh.data.vertices))).astype(int))
        groups = {}
        for vertex in mesh.data.vertices:
            for group in vertex.groups:
                if group.weight >= .25:
                    groups.setdefault(group.group, []).append(vertex.index)
        for values in groups.values():
            indices.update(values[i] for i in np.linspace(0, len(values) - 1, min(3, len(values))).astype(int))
        sample_indices[mesh.name] = sorted(indices)
    prefix_states = {}
    for value in frames:
        bpy.context.scene.frame_set(int(value), subframe=float(value) % 1)
        for mesh, rigs in static_prefixes:
            current = _deformation_state(mesh, rigs)
            previous = prefix_states.setdefault(mesh.name, current)
            if not np.allclose(current, previous, atol=1e-6, rtol=0):
                raise ValueError(f"{mesh.name}: the additional deformation rig moves during the clip. "
                                 "Bake or merge its animation before a single-skin export; use BLEND to retain the rig.")
        evaluated = armature.evaluated_get(bpy.context.evaluated_depsgraph_get())
        positions.append([tuple(evaluated.matrix_world @ evaluated.pose.bones[name].head) for name in names])
        world = evaluated.matrix_world
        inverse_world = world.inverted_safe()
        deformations.append([np.asarray((world @ evaluated.pose.bones[name].matrix @
            evaluated.data.bones[name].matrix_local.inverted_safe() @ inverse_world).to_3x3()) for name in names])
        morphs.append([bpy.data.objects[mesh].data.shape_keys.key_blocks[key].value for mesh, key in shape_names])
        current_surface = []
        graph = bpy.context.evaluated_depsgraph_get()
        for mesh in meshes:
            evaluated_mesh = mesh.evaluated_get(graph)
            evaluated_data = evaluated_mesh.to_mesh()
            try:
                if len(evaluated_data.vertices) != len(mesh.data.vertices):
                    raise ValueError("Bake topology-changing modifiers before verified skinned export: " + mesh.name)
                current_surface.extend(tuple(evaluated_mesh.matrix_world @ evaluated_data.vertices[i].co)
                                       for i in sample_indices[mesh.name])
            finally:
                evaluated_mesh.to_mesh_clear()
        surfaces.append(current_surface)
    return names, np.asarray(positions), shape_names, np.asarray(morphs), np.asarray(surfaces, np.float32), np.asarray(deformations)


@contextmanager
def _prepared_export_meshes(armature, meshes, start, end, validate):
    """Bake a static auxiliary armature prefix into temporary basis/morph geometry.

    The original evaluated stack is the reference. An animated auxiliary stack is
    rejected instead of silently dropping deformation from the exported character.
    """
    scene = bpy.context.scene
    frame, subframe = scene.frame_current, scene.frame_subframe
    copies, animation = [], []
    prefixes = []
    try:
        for obj in [armature, *meshes]:
            for owner in (obj, obj.data.shape_keys if obj.type == "MESH" else None):
                data = owner.animation_data if owner is not None else None
                if data:
                    animation.append((data, data.use_nla))
                    data.use_nla = False
        for mesh in meshes:
            enabled = [m for m in mesh.modifiers if m.show_viewport]
            auxiliary = [m for m in enabled if m.type == "ARMATURE" and m.object and m.object != armature]
            if not auxiliary:
                continue
            primary = next((i for i, m in enumerate(enabled) if m.type == "ARMATURE" and m.object == armature), None)
            if primary is None or any(m not in enabled[:primary] for m in auxiliary) or any(
                    m.type != "ARMATURE" for m in enabled[:primary]):
                raise ValueError(f"{mesh.name}: bake the auxiliary modifier stack before single-skin export.")
            prefixes.append((mesh, [m.object for m in auxiliary]))
        frames = np.arange(start, end + .01, .5)
        reference = _sample_pose(armature, meshes, frames, prefixes) if validate or prefixes else None
        scene.frame_set(start)
        for mesh, _ in prefixes:
            duplicate = mesh.copy()
            duplicate.data = mesh.data.copy()
            duplicate.name = mesh.name + "_ExportBake"
            scene.collection.objects.link(duplicate)
            copies.append(duplicate)
            primary = next(i for i, m in enumerate(duplicate.modifiers) if m.type == "ARMATURE" and m.object == armature)
            states = [(m, m.show_viewport) for m in duplicate.modifiers]
            for mod in list(duplicate.modifiers)[primary:]:
                mod.show_viewport = False
            keys = duplicate.data.shape_keys
            if keys and not keys.use_relative:
                raise ValueError("Bake absolute shape keys before exporting an auxiliary armature: " + mesh.name)
            key_animation = keys.animation_data if keys else None
            action = key_animation.action if key_animation else None
            action_slot = key_animation.action_slot if key_animation else None
            if key_animation:
                key_animation.action = None
                key_animation.use_nla = False
                for driver in list(key_animation.drivers):
                    keys.driver_remove(driver.data_path, driver.array_index)
            values = [(key, key.value, key.mute) for key in keys.key_blocks] if keys else []
            for key, _, _ in values:
                key.value = 0
                key.mute = False

            def evaluated_coordinates():
                bpy.context.view_layer.update()
                graph = bpy.context.evaluated_depsgraph_get()
                evaluated = duplicate.evaluated_get(graph)
                data = evaluated.to_mesh()
                try:
                    if len(data.vertices) != len(duplicate.data.vertices):
                        raise ValueError("Auxiliary export baking must preserve topology: " + mesh.name)
                    coordinates = np.empty(len(data.vertices) * 3, np.float32)
                    data.vertices.foreach_get("co", coordinates)
                    return coordinates
                finally:
                    evaluated.to_mesh_clear()

            basis = evaluated_coordinates()
            deltas = {}
            if keys:
                for key in list(keys.key_blocks)[1:]:
                    key.value = 1
                    deltas[key.name] = evaluated_coordinates() - basis
                    key.value = 0
                resolved = {keys.reference_key.name: basis}

                def coordinates_for(key):
                    if key.name not in resolved:
                        resolved[key.name] = coordinates_for(key.relative_key) + deltas[key.name]
                    return resolved[key.name]

                for key in keys.key_blocks:
                    key.data.foreach_set("co", coordinates_for(key))
                for key, value, muted in values:
                    key.value, key.mute = value, muted
                if key_animation:
                    key_animation.action = action
                    if action_slot:
                        key_animation.action_slot = action_slot
            duplicate.data.vertices.foreach_set("co", basis)
            duplicate.data.update()
            for mod, visible in states:
                if mod.type == "ARMATURE" and mod.object and mod.object != armature:
                    duplicate.modifiers.remove(mod)
                else:
                    mod.show_viewport = visible
        by_original = {mesh.name: copy for (mesh, _), copy in zip(prefixes, copies)}
        yield [by_original.get(mesh.name, mesh) for mesh in meshes], reference, len(copies)
    finally:
        for duplicate in copies:
            data = duplicate.data
            bpy.data.objects.remove(duplicate, do_unlink=True)
            if data.users == 0:
                bpy.data.meshes.remove(data)
        for data, use_nla in animation:
            data.use_nla = use_nla
        scene.frame_set(frame, subframe=subframe)


def _evaluated_surface(meshes):
    vertices, triangles, offset = [], [], 0
    graph = bpy.context.evaluated_depsgraph_get()
    for mesh in meshes:
        evaluated = mesh.evaluated_get(graph)
        data = evaluated.to_mesh()
        try:
            data.calc_loop_triangles()
            coordinates = np.empty(len(data.vertices)*3, np.float32)
            data.vertices.foreach_get("co", coordinates)
            matrix = np.asarray(evaluated.matrix_world)
            vertices.append(coordinates.reshape(-1, 3)@matrix[:3, :3].T+matrix[:3, 3])
            indices = np.empty(len(data.loop_triangles)*3, np.int32)
            data.loop_triangles.foreach_get("vertices", indices)
            triangles.append(indices.reshape(-1, 3)+offset)
            offset += len(data.vertices)
        finally:
            evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(np.concatenate(vertices).tolist(), np.concatenate(triangles).tolist(),
        all_triangles=True) if offset else None


def validate_roundtrip(path, armature, meshes, start, end, static=False, max_error_m=.003, source_auxiliary=(), reference=None):
    """Check all integer poses plus halfway poses against a fresh import scene."""
    context, original = bpy.context, bpy.context.scene
    if context.window is None:
        raise RuntimeError("A Blender window context is required for export validation.")
    frames = np.arange(start, end + .01, .5) if not static else np.array([start], float)
    # Compare with the original evaluated stack, including facial/control rigs.
    try:
        for modifier, rig in source_auxiliary:
            modifier.object = rig
        names, expected, shape_names, morphs, expected_surfaces, expected_deformations = reference if reference is not None else _sample_pose(armature, meshes, frames)
    finally:
        for modifier, _ in source_auxiliary:
            modifier.object = None
    new_scene = bpy.data.scenes.new("Nuranova_Export_Validation")
    new_scene.render.fps, new_scene.render.fps_base = original.render.fps, original.render.fps_base
    previous_objects = set(bpy.data.objects)
    report = {}
    try:
        context.window.scene = new_scene
        if path.suffix.lower() == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(path), use_anim=not static)
        else:
            bpy.ops.import_scene.gltf(filepath=str(path))
        imported = [obj for obj in new_scene.objects if obj.type == "ARMATURE"]
        if len(imported) != 1:
            raise ValueError(f"Expected one exported armature, found {len(imported)}.")
        actual_armature = imported[0]
        missing = [name for name in names if name not in actual_armature.pose.bones]
        if missing:
            raise ValueError("Export lost bones: " + ", ".join(missing[:8]))
        wrong_parents = [name for name in names if
            (armature.data.bones[name].parent.name if armature.data.bones[name].parent else None) !=
            (actual_armature.data.bones[name].parent.name if actual_armature.data.bones[name].parent else None)]
        if wrong_parents:
            raise ValueError("Export changed bone parents: " + ", ".join(wrong_parents[:8]))
        action = actual_armature.animation_data.action if actual_armature.animation_data else None
        if not static and end > start and action is None:
            raise ValueError("The exported file contains no armature animation.")
        offset = float(action.frame_range[0]) - start if action else 0.
        source_fps = original.render.fps / original.render.fps_base
        imported_fps = new_scene.render.fps / new_scene.render.fps_base
        # Blender 5.2's FBX reader (blen_read_animations) converts key times
        # with render.fps and omits fps_base. Compare in those key-time units;
        # the FBX itself retains seconds and the correct CustomFrameRate.
        keyframe_fps = float(new_scene.render.fps) if path.suffix.lower() == ".fbx" else imported_fps
        time_scale = keyframe_fps / source_fps
        actual, actual_deformations = [], []
        actual_morphs = []
        surface_errors = []
        imported_meshes = bound_meshes(actual_armature)
        expected_key_names = [key for _, key in shape_names]
        for sample_index, value in enumerate(frames):
            sample = start + offset + (float(value)-start)*time_scale
            new_scene.frame_set(int(np.floor(sample)), subframe=sample % 1)
            evaluated = actual_armature.evaluated_get(context.evaluated_depsgraph_get())
            actual.append([tuple(evaluated.matrix_world @ evaluated.pose.bones[name].head) for name in names])
            world = evaluated.matrix_world
            inverse_world = world.inverted_safe()
            actual_deformations.append([np.asarray((world @ evaluated.pose.bones[name].matrix @
                evaluated.data.bones[name].matrix_local.inverted_safe() @ inverse_world).to_3x3()) for name in names])
            current_morphs = []
            for key in expected_key_names:
                matches = [mesh.data.shape_keys.key_blocks[key].value for mesh in imported_meshes
                           if mesh.data.shape_keys and key in mesh.data.shape_keys.key_blocks]
                current_morphs.append(matches[0] if len(matches) == 1 else float("nan"))
            actual_morphs.append(current_morphs)
            if meshes:
                surface = _evaluated_surface(imported_meshes)
                distances = [surface.find_nearest(Vector(point))[3] for point in expected_surfaces[sample_index]] if surface else [float("inf")]
                surface_errors.append(max(distances, default=0.) * original.unit_settings.scale_length)
        error = np.linalg.norm(np.asarray(actual) - expected, axis=-1) * original.unit_settings.scale_length
        deformation_error = float(np.max(abs(np.asarray(actual_deformations) - expected_deformations)))
        duration = float(action.frame_range[1] - action.frame_range[0]) if action else 0
        normalized_duration = duration / time_scale
        report = {"passed": bool(np.max(error) <= max_error_m and deformation_error <= .001 and (static or abs(normalized_duration - (end - start)) <= 1.01)),
                  "samples": len(frames), "bones": len(names), "meshes": len(imported_meshes),
                  "max_joint_error_m": float(np.max(error)), "p95_joint_error_m": float(np.percentile(error, 95)),
                  "duration_frames": normalized_duration, "expected_duration_frames": end - start,
                  "imported_duration_frames": duration, "source_fps": source_fps, "imported_fps": imported_fps,
                  "duration_seconds": duration / keyframe_fps,
                  "import_keyframe_fps": keyframe_fps,
                  "blender_import_timebase_correction": keyframe_fps != imported_fps,
                  "frame_offset": offset, "shape_channels": len(shape_names)}
        report.update(hierarchy_verified=True, max_deformation_matrix_error=deformation_error)
        if surface_errors:
            report["max_surface_error_m"] = float(max(surface_errors))
            report["surface_samples_per_pose"] = len(expected_surfaces[0])
            report["passed"] = report["passed"] and max(surface_errors) <= max_error_m
        if shape_names:
            difference = abs(np.asarray(actual_morphs) - morphs)
            report["shape_channels_verified"] = bool(np.isfinite(difference).all() and np.max(difference) < .02)
            report["max_shape_error"] = float(np.nanmax(difference)) if np.isfinite(difference).any() else None
            report["passed"] = report["passed"] and report["shape_channels_verified"]
            if np.isfinite(difference).any():
                ti,ki=np.unravel_index(np.nanargmax(difference),difference.shape)
                report.update(max_shape_error_frame=float(frames[ti]),max_shape_error_channel=shape_names[ki][1])
        if meshes and not imported_meshes:
            report["passed"] = False
        return report
    finally:
        context.window.scene = original
        for obj in set(bpy.data.objects) - previous_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.scenes.remove(new_scene)


def export_character(armature, destination, *, variant="character", frame_start=None,
                     frame_end=None, fps=None, embed_textures=True, validate=True):
    """Deliver FBX/GLB/glTF with explicit scope. Publish only a verified newly written file."""
    if armature is None or armature.type != "ARMATURE":
        raise ValueError("Choose the target armature to export.")
    path = Path(destination).resolve()
    suffix = path.suffix.lower()
    if suffix not in {".fbx", ".glb", ".gltf"}:
        raise ValueError("Choose FBX, GLB or glTF.")
    if variant not in {"motion", "character", "static"}:
        raise ValueError("Invalid export content selection.")
    meshes = bound_meshes(armature) if variant != "motion" else []
    if variant == "character" and not meshes:
        raise ValueError("This target has no bound meshes. Choose motion-only or bind the character first.")
    start = int(frame_start if frame_start is not None else bpy.context.scene.frame_start)
    end = int(frame_end if frame_end is not None else bpy.context.scene.frame_end)
    if end < start:
        raise ValueError("The export range ends before it starts.")
    static = variant == "static"
    if static:
        start = int(frame_start if frame_start is not None else bpy.context.scene.frame_current)
        return export_static_character(armature, path, start, fps, embed_textures, validate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _prepared_export_meshes(armature, meshes, start, end, validate) as (prepared, reference, baked_count), tempfile.TemporaryDirectory(prefix="nuranova_export_", dir=path.parent) as folder:
        staging = Path(folder) / path.name
        with export_scope(armature, prepared, start, end, fps, static) as auxiliary, _fbx_motion_bind_pose(suffix == ".fbx" and variant == "motion"):
            if suffix == ".fbx":
                bpy.ops.export_scene.fbx(filepath=str(staging), check_existing=False, use_selection=True,
                    object_types={"ARMATURE", "MESH"}, bake_anim=not static,
                    bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                    bake_anim_simplify_factor=0., bake_anim_step=1., add_leaf_bones=False,
                    use_armature_deform_only=False, use_mesh_modifiers=False,
                    path_mode="COPY" if embed_textures else "ABSOLUTE", embed_textures=embed_textures)
            else:
                bpy.ops.export_scene.gltf(filepath=str(staging), check_existing=False,
                    export_format="GLB" if suffix == ".glb" else "GLTF_SEPARATE", use_selection=True,
                    use_active_scene=True,
                    export_animations=not static, export_animation_mode="SCENE", export_frame_range=True,
                    export_anim_scene_split_object=False,
                    export_frame_step=1, export_force_sampling=True, export_optimize_animation_size=False,
                    export_skins=True, export_all_influences=True, export_morph=True,
                    export_morph_animation=not static, export_cameras=False, export_lights=False,
                    export_apply=False, export_current_frame=static,
                    export_morph_reset_sk_data=False, export_reset_pose_bones=False,
                    export_image_format="AUTO" if embed_textures else "NONE")
            if not staging.exists() or staging.stat().st_size < 100:
                raise RuntimeError("Blender did not produce a complete export.")
            report = validate_roundtrip(staging, armature, prepared, start, end, static, source_auxiliary=auxiliary,reference=reference) if validate else {"passed": None, "status": "not validated"}
            if validate and not report["passed"]:
                raise ValueError("Export round-trip validation failed: " + json.dumps(report))
        _publish_export(staging, path)
    report.update(file=str(path), variant=variant, target=armature.name, static_auxiliary_meshes_baked=baked_count,
                  bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  fps=bpy.context.scene.render.fps / bpy.context.scene.render.fps_base)
    path.with_suffix(path.suffix + ".validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _publish_export(staging, path):
        if path.suffix.lower() == ".gltf":
            import shutil
            payload = json.loads(staging.read_text(encoding="utf-8"))
            resource_dir = path.parent / (path.stem + "_resources")
            resource_dir.mkdir(exist_ok=True)
            from urllib.parse import quote, unquote
            for entry in payload.get("buffers", []) + payload.get("images", []):
                uri = entry.get("uri", "")
                if uri and not uri.startswith(("data:", "http:", "https:")):
                    original = staging.parent / unquote(uri)
                    relative = original.relative_to(staging.parent)
                    target = resource_dir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, target)
                    entry["uri"] = quote(resource_dir.name + "/" + relative.as_posix())
            staging.write_text(json.dumps(payload), encoding="utf-8")
        staging.replace(path)


def export_static_character(armature, path, frame, fps, textures, validate):
    scene, context = bpy.context.scene, bpy.context
    previous_frame = scene.frame_current
    copies = []
    scene.frame_set(frame)
    try:
        graph = context.evaluated_depsgraph_get()
        for obj in bound_meshes(armature):
            evaluated = obj.evaluated_get(graph)
            data = bpy.data.meshes.new_from_object(evaluated, depsgraph=graph)
            copy = bpy.data.objects.new(obj.name + "_StaticExport", data)
            scene.collection.objects.link(copy)
            copy.matrix_world = evaluated.matrix_world.copy()
            copies.append(copy)
        if not copies:
            raise ValueError("A static character export requires bound meshes.")
        expected = _sample_pose(armature, copies, [frame])[4][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="nuranova_static_", dir=path.parent) as folder:
            staging = Path(folder) / path.name
            with export_scope(armature, copies, frame, frame, fps, static=True, include_armature=False):
                if path.suffix.lower() == ".fbx":
                    bpy.ops.export_scene.fbx(filepath=str(staging), check_existing=False, use_selection=True,
                        object_types={"MESH"}, bake_anim=False, path_mode="COPY" if textures else "ABSOLUTE", embed_textures=textures)
                else:
                    bpy.ops.export_scene.gltf(filepath=str(staging), check_existing=False, use_selection=True,
                        use_active_scene=True,
                        export_format="GLB" if path.suffix.lower() == ".glb" else "GLTF_SEPARATE",
                        export_animations=False, export_skins=False, export_current_frame=True,
                        export_image_format="AUTO" if textures else "NONE")
                report = {"passed":None, "samples":1, "bones":0, "meshes":len(copies), "max_joint_error_m":0.}
                if validate:
                    imported_scene = bpy.data.scenes.new("Static_Export_Validation")
                    old_objects = set(bpy.data.objects)
                    try:
                        context.window.scene = imported_scene
                        if path.suffix.lower() == ".fbx":
                            bpy.ops.import_scene.fbx(filepath=str(staging), use_anim=False)
                        else:
                            bpy.ops.import_scene.gltf(filepath=str(staging))
                        meshes = [o for o in imported_scene.objects if o.type == "MESH"]
                        surface = _evaluated_surface(meshes)
                        error = max(surface.find_nearest(Vector(p))[3] for p in expected) * scene.unit_settings.scale_length if surface else float("inf")
                        report.update(passed=bool(error <= .003 and len(meshes) == len(copies)), max_surface_error_m=error)
                    finally:
                        context.window.scene = scene
                        for obj in set(bpy.data.objects) - old_objects:
                            bpy.data.objects.remove(obj, do_unlink=True)
                        bpy.data.scenes.remove(imported_scene)
                    if not report["passed"]:
                        raise ValueError("Static mesh round-trip verification failed: " + json.dumps(report))
            _publish_export(staging, path)
        report.update(file=str(path), variant="static", target=armature.name, bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(), fps=fps)
        path.with_suffix(path.suffix + ".validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        for obj in copies:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data.users == 0:
                bpy.data.meshes.remove(data)
        scene.frame_set(previous_frame)
