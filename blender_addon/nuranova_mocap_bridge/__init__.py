"""Nuranova Mocap Bridge v1.3.2 — Blender 4.5 LTS / 5.0+ Extension Entry Point.

License: GPL-3.0-or-later (per Blender Python API `bpy` licensing requirements).
Features:
- Smart Landmark Marker Auto-Rig (Guided Click-to-Place with automatic through-mesh depth centering,
  X-symmetry mirroring, live 3D skeleton wireframe preview, optional finger markers & iterative re-bind —
  works on stylized, anime, cartoon, and realistic characters of ANY proportion!)
- Real-Time Live Mocap Preview directly in Blender's 3D Viewport (25 body + 30 finger bones!)
- Automatic facing-yaw alignment + Rig Presets (Nuranova, Rigify, Mixamo, Unreal, VRM/Unity)
- Editable quaternion keyframe baking, NLA tracks, 52 Facial Shape Keys, and FBX / BVH export.
"""

from __future__ import annotations

import json
import queue
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from mathutils import Vector

from .auto_rig import (
    auto_rig_character_mesh,
    auto_rig_from_markers,
    create_nuranova_humanoid_rig,
)
from .facial_hands import (
    apply_facial_blendshapes_to_mesh,
    apply_hand_rotations_to_armature,
    apply_live_hand_rotations,
)
from .retargeting import (
    auto_match_bones,
    load_mapping_profile,
    retarget_and_bake,
    retarget_live_frame,
    save_mapping_profile,
)
from .quality_ops import QUALITY_CLASSES, draw_quality_panel
from .smart_markers import (
    BODY_MARKER_SPECS,
    MARKER_COLLECTION_NAME,
    MARKER_PREFIX,
    PREVIEW_MARKER_EDGES,
    clear_markers,
    collect_marker_positions,
    compute_meshes_world_bbox,
    count_placed_body_landmarks,
    get_symmetry_center_x,
    mirror_markers_lr,
    place_body_landmark,
    raycast_mesh_center,
    set_markers_visibility,
    spawn_finger_markers,
    spawn_preset_markers,
)
from .source_armature import (
    apply_live_frame_to_source_armature,
    create_live_preview_armature,
    create_source_armature_from_payload,
    load_nmocap_json,
)

bl_info = {
    "name": "Nuranova Mocap Bridge",
    "author": "Nuranova Solutions (Kawshal, Shanal, Shehan)",
    "version": (1, 3, 2),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Nuranova Mocap",
    "description": "Smart Marker Auto-Rig (Body + 30 Fingers), Live Mocap Preview, Auto-Retarget, Bake & Export for Nuranova Mocap Studio",
    "category": "Animation",
}

# Thread-safe mailbox for main-thread scene updates via bpy.app.timers
_BRIDGE_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
_LAST_PAYLOAD_CACHE: Dict[str, Any] = {}
_LIVE_FETCH_IN_FLIGHT = threading.Event()
_LAST_LIVE_SEQ: int = -1

# Persistent 3D skeleton wireframe draw handler for visible Smart Markers
_PREVIEW_3D_HANDLER: Any = None


def _draw_smart_marker_skeleton_3d() -> None:
    """Draw a live X-ray cyan skeleton preview connecting all visible Smart Markers in the 3D Viewport."""
    if bpy.app.background:
        return
    try:
        col = bpy.data.collections.get(MARKER_COLLECTION_NAME)
        if col is None:
            return
        vis_positions: Dict[str, Vector] = {}
        for obj in col.objects:
            if obj.hide_viewport or obj.hide_get():
                continue
            key = obj.get("nuranova_marker_key")
            if not key and obj.name.startswith(MARKER_PREFIX):
                key = obj.name[len(MARKER_PREFIX):]
            if key:
                vis_positions[str(key)] = obj.matrix_world.translation.copy()
        if len(vis_positions) < 2:
            return

        body_lines: List[Tuple[float, float, float]] = []
        for a, b in PREVIEW_MARKER_EDGES:
            if a in vis_positions and b in vis_positions:
                body_lines.append(tuple(vis_positions[a]))
                body_lines.append(tuple(vis_positions[b]))

        finger_lines: List[Tuple[float, float, float]] = []
        for suf in ("L", "R"):
            wrist = vis_positions.get(f"wrist_{suf}")
            for fname in ("thumb", "index", "middle", "ring", "pinky"):
                kb = f"f_{fname}_base_{suf}"
                kt = f"f_{fname}_tip_{suf}"
                if kb in vis_positions and kt in vis_positions:
                    if wrist is not None:
                        finger_lines.append(tuple(wrist))
                        finger_lines.append(tuple(vis_positions[kb]))
                    finger_lines.append(tuple(vis_positions[kb]))
                    finger_lines.append(tuple(vis_positions[kt]))

        if not body_lines and not finger_lines:
            return

        import gpu
        from gpu_extras.batch import batch_for_shader

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        if body_lines:
            gpu.state.line_width_set(3.0)
            batch = batch_for_shader(shader, "LINES", {"pos": body_lines})
            shader.bind()
            shader.uniform_float("color", (0.10, 0.92, 0.86, 0.92))
            batch.draw(shader)
        if finger_lines:
            gpu.state.line_width_set(2.0)
            batch_f = batch_for_shader(shader, "LINES", {"pos": finger_lines})
            shader.bind()
            shader.uniform_float("color", (0.98, 0.48, 1.00, 0.92))
            batch_f.draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.blend_set("NONE")
    except Exception:
        pass


def _resolve_target_character_meshes(context: bpy.types.Context, props: Any) -> List[bpy.types.Object]:
    """Resolve character mesh(es) from current selection, Character Mesh picker, or remembered marker targets."""
    meshes: List[bpy.types.Object] = []
    seen = set()

    def _add(o: Optional[bpy.types.Object]) -> None:
        if o is not None and o.type == "MESH" and o.name in bpy.data.objects and o.name not in seen:
            seen.add(o.name)
            meshes.append(o)

    # 1. Currently selected non-marker meshes (supports multi-mesh characters: body + hair + clothes!)
    for o in context.selected_objects:
        if o.type == "MESH" and not o.name.startswith(MARKER_PREFIX):
            _add(o)

    # 2. Explicit Character Mesh property
    if props.target_character_mesh is not None:
        _add(props.target_character_mesh)

    # 3. Remembered mesh names from when user placed Smart Markers
    if not meshes and props.smart_rig_mesh_names:
        for nm in props.smart_rig_mesh_names.split("||"):
            nm = nm.strip()
            if nm and nm in bpy.data.objects:
                _add(bpy.data.objects[nm])

    # 4. Active mesh fallback
    if not meshes and context.active_object is not None and context.active_object.type == "MESH":
        _add(context.active_object)

    return meshes


def _remember_target_meshes(props: Any, meshes: List[bpy.types.Object]) -> None:
    if meshes:
        if props.target_character_mesh is None:
            props.target_character_mesh = meshes[0]
        props.smart_rig_mesh_names = "||".join(m.name for m in meshes)


def _tag_all_view3d_redraw(context: bpy.types.Context) -> None:
    if context.screen is None:
        return
    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()


def _mapping_contains_fingers(mapping: Dict[str, str]) -> bool:
    return any(
        any(f in k.lower() for f in ("thumb", "index", "middle", "ring", "pinky"))
        for k in mapping
    )


def _fetch_json_async(url: str, tag: str, timeout: float = 4.0) -> None:
    def worker() -> None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NuranovaMocapBridge/1.2"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            _BRIDGE_QUEUE.put({"tag": tag, "ok": True, "data": data})
        except Exception as exc:
            _BRIDGE_QUEUE.put({"tag": tag, "ok": False, "error": str(exc)})
        finally:
            if tag == "live_pose":
                _LIVE_FETCH_IN_FLIGHT.clear()

    threading.Thread(target=worker, daemon=True).start()


def _populate_bone_map_collection(props: Any, mapping: Dict[str, str]) -> None:
    props.bone_map_items.clear()
    for s, t in mapping.items():
        item = props.bone_map_items.add()
        item.source_bone = s
        item.target_bone = t


def _apply_live_pose_message(context: bpy.types.Context, props: Any, data: Dict[str, Any]) -> None:
    global _LAST_LIVE_SEQ
    if not data.get("active"):
        props.live_status_text = "Live waiting — start Live Preview or Mocap in Studio"
        return

    seq = int(data.get("seq", 0))
    if seq == _LAST_LIVE_SEQ:
        return
    _LAST_LIVE_SEQ = seq

    src_arm = props.source_armature
    if src_arm is None or src_arm.name not in bpy.data.objects:
        src_arm = create_live_preview_armature(
            data,
            armature_name="Nuranova_Live_Preview",
            include_fingers=props.include_fingers,
            context=context,
        )
        props.source_armature = src_arm

    apply_live_frame_to_source_armature(
        src_arm,
        data,
        in_place=props.in_place,
        include_fingers=props.include_fingers,
    )

    tgt_arm = props.target_armature
    if tgt_arm is not None and tgt_arm.name in bpy.data.objects and tgt_arm != src_arm:
        mapping = {it.source_bone: it.target_bone for it in props.bone_map_items if it.source_bone and it.target_bone}
        if not mapping:
            mapping = auto_match_bones(src_arm, tgt_arm, rig_preset=props.rig_preset)
            _populate_bone_map_collection(props, mapping)
        if mapping:
            retarget_live_frame(
                source_arm=src_arm,
                target_arm=tgt_arm,
                bone_map=mapping,
                scale_compensation=props.scale_compensation,
                extract_root_motion=props.extract_root_motion,
                in_place=props.in_place,
                ground_to_floor=props.ground_to_floor,
                context=context,
            )
        # Only fall back to direct finger rotation write if finger bones weren't already retargeted via `mapping`
        if props.include_fingers and data.get("hands") and not _mapping_contains_fingers(mapping):
            apply_live_hand_rotations(tgt_arm, data["hands"], include_wrist=False)

    hands_info = data.get("hands") or {}
    f_tag = ""
    if hands_info.get("left_detected") or hands_info.get("right_detected"):
        f_tag = " · Fingers OK"
    props.live_status_text = f"LIVE {data.get('fps', 0):.0f} FPS · {data.get('source', 'Studio')}{f_tag}"


def _bridge_main_thread_timer() -> float:
    """Runs safely on Blender's main thread to consume Studio HTTP & Live Preview responses."""
    next_interval = 0.25
    try:
        scene = bpy.context.scene
        props = getattr(scene, "nuranova_mocap", None) if scene is not None else None

        while not _BRIDGE_QUEUE.empty():
            msg = _BRIDGE_QUEUE.get_nowait()
            if props is None:
                continue
            tag = msg["tag"]
            if tag == "status":
                if msg["ok"]:
                    d = msg["data"]
                    props.studio_connected = True
                    props.studio_status_text = f"Connected ({d.get('active_project_name') or 'Studio'}, {len(d.get('available_takes', []))} takes)"
                    takes = d.get("available_takes", [])
                    if takes:
                        props.selected_take_id = str(takes[-1]["take_id"])
                else:
                    props.studio_connected = False
                    props.studio_status_text = f"Disconnected: {msg['error']}"
            elif tag == "import_take":
                if msg["ok"]:
                    payload = msg["data"]
                    _LAST_PAYLOAD_CACHE["latest"] = payload
                    arm = create_source_armature_from_payload(
                        payload,
                        in_place=props.in_place,
                        push_nla=props.push_to_nla,
                        include_fingers=props.include_fingers,
                        context=bpy.context,
                    )
                    props.source_armature = arm
                    if props.target_armature is not None:
                        mapping = auto_match_bones(arm, props.target_armature, rig_preset=props.rig_preset)
                        _populate_bone_map_collection(props, mapping)
                    props.studio_status_text = f"Imported take '{payload.get('take_id')}' ({payload.get('num_frames')} frames)"
                else:
                    props.studio_status_text = f"Import failed: {msg['error']}"
            elif tag == "live_pose":
                if msg["ok"] and props.live_preview_active:
                    _apply_live_pose_message(bpy.context, props, msg["data"])
                elif not msg["ok"] and props.live_preview_active:
                    props.live_status_text = f"Live error: {msg['error']}"

        if props is not None and props.live_preview_active:
            next_interval = 0.055
            if not _LIVE_FETCH_IN_FLIGHT.is_set():
                _LIVE_FETCH_IN_FLIGHT.set()
                _fetch_json_async(f"{props.studio_url.rstrip('/')}/api/v1/bridge/live", "live_pose", timeout=0.9)
    except Exception:
        pass
    return next_interval


class NuranovaBoneMapItem(bpy.types.PropertyGroup):
    source_bone: StringProperty(name="Source Bone")
    target_bone: StringProperty(name="Target Bone")


class NuranovaMocapProperties(bpy.types.PropertyGroup):
    studio_url: StringProperty(name="Studio URL", default="http://127.0.0.1:8765")
    studio_connected: BoolProperty(name="Connected", default=False)
    studio_status_text: StringProperty(name="Status", default="Not connected")
    selected_take_id: StringProperty(name="Take ID", default="")

    # Live Preview in Blender
    live_preview_active: BoolProperty(name="Live Preview Active", default=False)
    live_status_text: StringProperty(name="Live Status", default="Live preview idle")

    # Auto-Rig & Smart Marker Options
    rig_preset: EnumProperty(
        name="Rig Preset",
        description="Target character skeleton convention for auto bone & finger mapping",
        items=[
            ("AUTO", "Auto-Detect", "Automatically detect Rigify, Mixamo, Unreal, VRM, or Nuranova bones"),
            ("NURANOVA", "Nuranova Full (55 Bones)", "Nuranova Auto-Rig / Source Rig (25 body + 30 fingers)"),
            ("RIGIFY", "Rigify (DEF / f_index)", "Blender Rigify deform & finger bones"),
            ("MIXAMO", "Mixamo", "Adobe Mixamo character skeleton (body + fingers)"),
            ("UNREAL", "Unreal Mannequin", "UE4 / UE5 Mannequin skeleton"),
            ("VRM", "VRM / Unity Humanoid", "VRM / Unity Humanoid bones"),
        ],
        default="AUTO",
    )
    include_fingers: BoolProperty(
        name="Include 30 Finger Bones",
        description="Create, stream, and retarget all 30 articulated finger bones (left & right hands)",
        default=True,
    )
    mirror_markers: BoolProperty(
        name="Mirror L/R Markers (X-Symmetry)",
        description="Automatically mirror shoulder/elbow/wrist/hand/hip/knee/ankle/toe across X when placing markers",
        default=True,
    )
    character_facing: EnumProperty(
        name="Character Facing",
        description="Direction the character faces in Blender (Auto detects from Toe vs Ankle, default -Y Front View)",
        items=[
            ("AUTO", "Auto (Detect from Toes / -Y)", "Automatically detect facing direction from foot/toe markers"),
            ("NEG_Y", "Front -Y (Blender Standard)", "Character faces -Y (standard Numpad 1 Front View)"),
            ("POS_Y", "Back +Y", "Character faces +Y"),
        ],
        default="AUTO",
    )
    smart_rig_mesh_names: StringProperty(name="Smart Rig Meshes", default="")
    guided_marker_prompt: StringProperty(name="Marker Prompt", default="")

    auto_rig_height_m: FloatProperty(name="Rig Height (m)", default=1.72, min=0.4, max=3.5)
    target_character_mesh: PointerProperty(
        name="Character Mesh",
        description="Primary unrigged character mesh (you can also select body + hair + clothes together!)",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )

    motion_filepath: StringProperty(name="Motion File", subtype="FILE_PATH", default="")
    mapping_filepath: StringProperty(name="Mapping Profile", subtype="FILE_PATH", default="")

    source_armature: PointerProperty(
        name="Source Rig",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE",
    )
    target_armature: PointerProperty(
        name="Target Character Rig",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE",
    )
    target_face_mesh: PointerProperty(
        name="Target Face Mesh",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )

    bone_map_items: CollectionProperty(type=NuranovaBoneMapItem)
    bone_map_index: IntProperty(default=0)

    scale_compensation: BoolProperty(name="Scale Compensation", default=True)
    extract_root_motion: BoolProperty(name="Extract Root Motion", default=True)
    ground_to_floor: BoolProperty(
        name="Ground Feet to Floor (Z=0)",
        description="Automatically pin planted feet flush to the character's rest floor height (eliminates floating)",
        default=True,
    )
    in_place: BoolProperty(name="In-Place Animation", default=False)
    push_to_nla: BoolProperty(name="Push to NLA Track", default=True)
    face_gain: FloatProperty(name="Facial Blendshape Gain", default=1.0, min=0.0, max=3.0)
    export_filepath: StringProperty(name="Export Path", subtype="FILE_PATH", default="//nuranova_export.fbx")
    export_variant: EnumProperty(name="Content", items=[("character", "Character + animation", "Bound target meshes and active motion"),
        ("motion", "Motion only", "Armature animation"), ("static", "Static pose", "Current pose")], default="character")
    quality_start: IntProperty(name="Start (0 = scene)", default=0, min=0)
    quality_end: IntProperty(name="End (0 = scene)", default=0, min=0)
    quality_margin_mm: FloatProperty(name="Clearance (mm)", default=3., min=0., max=30.)
    quality_status: StringProperty(default="Analyze the selected character's final deformed meshes.")
    quality_original: PointerProperty(type=bpy.types.Object)
    quality_working: PointerProperty(type=bpy.types.Object)
    quality_contact_pair: StringProperty(name="Contact regions", default="left_hand,torso")
    quality_contact_kind: EnumProperty(name="Intent", items=[("touch", "Touch", "Preserve contact, correct penetration"),
                                                            ("pin", "Pin", "Preserve this endpoint; report infeasible contacts")])


# ---------------- Smart Landmark Marker Operators ----------------

class NURANOVA_OT_guided_place_markers(bpy.types.Operator):
    """Interactive Guided Click-to-Place Smart Marker Tool (with internal depth centering & X-mirror)."""

    bl_idname = "nuranova.guided_place_markers"
    bl_label = "Start Guided Marker Placement"
    bl_description = (
        "Click 12 guided landmarks directly on your character (auto-centers inside limbs & mirrors L/R). "
        "Middle-Mouse to orbit/zoom, Backspace to undo, ESC to stop"
    )

    _steps: List[Tuple[str, str, str, Optional[str]]] = []
    _step_idx: int = 0
    _mesh_names: List[str] = []
    _center_x: float = 0.0
    _char_height: float = 1.72
    _bbox_center: Vector = Vector((0.0, 0.0, 0.9))
    _hud_handler: Any = None
    _placed_history: List[List[str]] = []

    def _draw_hud_2d(self) -> None:
        if bpy.app.background or self._step_idx >= len(self._steps):
            return
        try:
            region = bpy.context.region
            if region is None:
                return
            import blf
            import gpu
            from gpu_extras.batch import batch_for_shader

            _key, label, hint, side = self._steps[self._step_idx]
            step_num = self._step_idx + 1
            total = len(self._steps)
            side_txt = f" ({side})" if side else ""
            title = f"NURANOVA SMART RIG  [{step_num}/{total}]  CLICK: {label.upper()}{side_txt}"
            sub = f"{hint}   |   MMB: Orbit/Zoom  ·  Backspace: Undo  ·  ESC: Finish"

            w = region.width
            h = region.height
            box_w = min(max(580, w - 120), 920)
            x0 = 0.5 * (w - box_w)
            x1 = x0 + box_w
            y1 = h - 18
            y0 = y1 - 68

            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            gpu.state.blend_set("ALPHA")
            verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]
            batch = batch_for_shader(shader, "TRIS", {"pos": verts})
            shader.bind()
            shader.uniform_float("color", (0.04, 0.08, 0.13, 0.88))
            batch.draw(shader)

            # Teal top accent bar
            bar_verts = [(x0, y1 - 4), (x1, y1 - 4), (x1, y1), (x0, y1 - 4), (x1, y1), (x0, y1)]
            batch_bar = batch_for_shader(shader, "TRIS", {"pos": bar_verts})
            shader.bind()
            shader.uniform_float("color", (0.10, 0.88, 0.82, 1.0))
            batch_bar.draw(shader)
            gpu.state.blend_set("NONE")

            blf.size(0, 17)
            blf.color(0, 0.18, 0.96, 0.90, 1.0)
            blf.position(0, x0 + 18, y1 - 31, 0)
            blf.draw(0, title)

            blf.size(0, 13)
            blf.color(0, 0.92, 0.95, 0.98, 1.0)
            blf.position(0, x0 + 18, y0 + 13, 0)
            blf.draw(0, sub)
        except Exception:
            pass

    def _update_header(self, context: bpy.types.Context) -> None:
        props = context.scene.nuranova_mocap
        if self._step_idx < len(self._steps):
            _key, label, hint, side = self._steps[self._step_idx]
            s_txt = f" ({side})" if side else ""
            msg = f"Step {self._step_idx + 1}/{len(self._steps)} — Click: {label}{s_txt} ({hint})"
            props.guided_marker_prompt = msg
            if context.area:
                context.area.header_text_set(f"{msg}  |  Backspace: Undo  |  ESC: Finish")
        else:
            props.guided_marker_prompt = "All landmarks placed! Adjust any marker or click 'Generate 55-Bone Rig & Bind'."
            if context.area:
                context.area.header_text_set(None)

    def _cleanup(self, context: bpy.types.Context) -> None:
        if self._hud_handler is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._hud_handler, "WINDOW")
            except Exception:
                pass
            self._hud_handler = None
        try:
            context.window.cursor_modal_restore()
        except Exception:
            pass
        if context.area:
            context.area.header_text_set(None)
        _tag_all_view3d_redraw(context)

    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"ERROR"}, "Run Guided Marker Placement from inside the 3D Viewport.")
            return {"CANCELLED"}

        props = context.scene.nuranova_mocap
        meshes = _resolve_target_character_meshes(context, props)
        if not meshes:
            self.report({"ERROR"}, "Select your character Mesh object (or pick Character Mesh above) first.")
            return {"CANCELLED"}

        _remember_target_meshes(props, meshes)
        self._mesh_names = [m.name for m in meshes]
        bmin, bmax = compute_meshes_world_bbox(meshes)
        self._char_height = max(0.25, float(bmax.z - bmin.z))
        self._center_x = 0.5 * float(bmin.x + bmax.x)
        self._bbox_center = 0.5 * (bmin + bmax)

        # Build guided landmark sequence (12 steps with X-mirror, 20 steps without mirror)
        self._steps = []
        for spec in BODY_MARKER_SPECS:
            if not spec.symmetric or props.mirror_markers:
                self._steps.append((spec.key, spec.label, spec.hint, None))
            else:
                self._steps.append((spec.key, f"{spec.label} (LEFT side)", f"Click the character's LEFT {spec.label.lower()}", "L"))
                self._steps.append((spec.key, f"{spec.label} (RIGHT side)", f"Click the character's RIGHT {spec.label.lower()}", "R"))

        self._step_idx = 0
        self._placed_history = []
        set_markers_visibility(True)

        if not bpy.app.background:
            self._hud_handler = bpy.types.SpaceView3D.draw_handler_add(
                self._draw_hud_2d, (), "WINDOW", "POST_PIXEL"
            )
            try:
                context.window.cursor_modal_set("CROSSHAIR")
            except Exception:
                pass

        self._update_header(context)
        context.window_manager.modal_handler_add(self)
        _tag_all_view3d_redraw(context)
        return {"RUNNING_MODAL"}

    def _click_is_in_view3d_canvas(self, context: bpy.types.Context, event: bpy.types.Event) -> Tuple[bool, Optional[bpy.types.Region], Optional[bpy.types.RegionView3D]]:
        area = context.area
        if area is None or area.type != "VIEW_3D":
            return False, None, None
        mx, my = event.mouse_x, event.mouse_y
        # Reject clicks over N-panel sidebar, toolbar, or header so UI buttons still work
        for reg in area.regions:
            if reg.type != "WINDOW":
                if reg.x <= mx <= reg.x + reg.width and reg.y <= my <= reg.y + reg.height:
                    return False, None, None
        for reg in area.regions:
            if reg.type == "WINDOW":
                if reg.x <= mx <= reg.x + reg.width and reg.y <= my <= reg.y + reg.height:
                    return True, reg, area.spaces.active.region_3d
        return False, None, None

    def modal(self, context, event):
        _tag_all_view3d_redraw(context)

        # Let user freely orbit, pan, zoom, and switch Front/Side/Top views while marking!
        if event.type in {
            "MIDDLEMOUSE",
            "WHEELUPMOUSE",
            "WHEELDOWNMOUSE",
            "TRACKPADPAN",
            "TRACKPADZOOM",
            "NUMPAD_1",
            "NUMPAD_3",
            "NUMPAD_7",
            "NUMPAD_5",
            "NUMPAD_PERIOD",
            "LEFT_ALT",
            "RIGHT_ALT",
            "LEFT_SHIFT",
            "RIGHT_SHIFT",
            "MOUSEMOVE",
            "INBETWEEN_MOUSEMOVE",
        }:
            return {"PASS_THROUGH"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self._cleanup(context)
            placed, total = count_placed_body_landmarks()
            context.scene.nuranova_mocap.guided_marker_prompt = f"Stopped ({placed}/{total} body landmarks placed)."
            self.report({"INFO"}, f"Marker placement paused ({placed}/{total} landmarks placed).")
            return {"FINISHED"}

        if event.type in {"BACK_SPACE", "Z"} and event.value == "PRESS":
            if self._step_idx > 0 and self._placed_history:
                self._step_idx -= 1
                prev_names = self._placed_history.pop()
                for nm in prev_names:
                    obj = bpy.data.objects.get(nm)
                    if obj is not None:
                        bpy.data.objects.remove(obj, do_unlink=True)
                self._update_header(context)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            in_canvas, region, rv3d = self._click_is_in_view3d_canvas(context, event)
            if not in_canvas or region is None or rv3d is None:
                return {"PASS_THROUGH"}

            from bpy_extras import view3d_utils

            coord = (event.mouse_x - region.x, event.mouse_y - region.y)
            ray_orig = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            ray_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            meshes = [bpy.data.objects[nm] for nm in self._mesh_names if nm in bpy.data.objects]

            hit_pos, _hit_ok = raycast_mesh_center(
                context=context,
                mesh_objs=meshes,
                ray_origin=ray_orig,
                ray_dir=ray_dir,
                fallback_plane_point=self._bbox_center,
                max_thickness_m=self._char_height * 0.42,
            )

            props = context.scene.nuranova_mocap
            spec_key, _label, _hint, explicit_side = self._steps[self._step_idx]
            # Refine symmetry center dynamically once center spine markers are placed
            cx = get_symmetry_center_x(mesh_objs=meshes, existing_markers=collect_marker_positions())
            placed_objs = place_body_landmark(
                spec_key=spec_key,
                world_pos=hit_pos,
                center_x=cx,
                char_height=self._char_height,
                mirror=props.mirror_markers,
                facing_override=props.character_facing,
                explicit_side=explicit_side,
                context=context,
            )
            self._placed_history.append([o.name for o in placed_objs])
            self._step_idx += 1
            self._update_header(context)

            if self._step_idx >= len(self._steps):
                self._cleanup(context)
                self.report(
                    {"INFO"},
                    "All 12 Smart Landmarks placed! Orbit to verify (drag any marker with G) then click 'Generate 55-Bone Rig & Bind'.",
                )
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}


class NURANOVA_OT_spawn_preset_markers(bpy.types.Operator):
    bl_idname = "nuranova.spawn_preset_markers"
    bl_label = "Place Preset Markers (Drag to Fit)"
    bl_description = "Spawn all 20 named body landmark markers around your character so you can drag them with G"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        meshes = _resolve_target_character_meshes(context, props)
        if not meshes:
            self.report({"ERROR"}, "Select your character Mesh object (or pick Character Mesh above) first.")
            return {"CANCELLED"}
        _remember_target_meshes(props, meshes)
        created = spawn_preset_markers(meshes, facing_override=props.character_facing, context=context)
        props.guided_marker_prompt = "Preset markers spawned! Drag with G in Front/Side view, then Generate Rig."
        _tag_all_view3d_redraw(context)
        self.report({"INFO"}, f"Spawned {len(created)} Smart Markers on '{meshes[0].name}'. Move any marker with G, then click Generate Rig!")
        return {"FINISHED"}


class NURANOVA_OT_spawn_finger_markers(bpy.types.Operator):
    bl_idname = "nuranova.spawn_finger_markers"
    bl_label = "Add Optional Finger Markers"
    bl_description = "Spawn 20 fine-tuning knuckle & tip markers along both hands for surgical finger placement"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        created = spawn_finger_markers(facing_override=props.character_facing, context=context)
        if not created:
            self.report({"WARNING"}, "Place Wrist and Hand Tip body markers first.")
            return {"CANCELLED"}
        _tag_all_view3d_redraw(context)
        self.report({"INFO"}, f"Added {len(created)} finger markers! Drag them onto knuckles/tips, Mirror L<->R, then Generate Rig.")
        return {"FINISHED"}


class NURANOVA_OT_mirror_markers(bpy.types.Operator):
    bl_idname = "nuranova.mirror_markers"
    bl_label = "Mirror L <-> R"
    bl_description = "Mirror Left markers to Right (or Right to Left if an _R marker is active)"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        meshes = _resolve_target_character_meshes(context, props)
        n = mirror_markers_lr(mesh_objs=meshes, source_side="AUTO", context=context)
        _tag_all_view3d_redraw(context)
        self.report({"INFO"}, f"Mirrored {n} symmetric markers across X center.")
        return {"FINISHED"}


class NURANOVA_OT_toggle_markers(bpy.types.Operator):
    bl_idname = "nuranova.toggle_markers"
    bl_label = "Show / Hide Markers"
    bl_description = "Toggle visibility of Smart Rig Markers so you can refine positions and re-generate anytime"

    def execute(self, context):
        col = bpy.data.collections.get(MARKER_COLLECTION_NAME)
        if col is None or len(col.objects) == 0:
            self.report({"WARNING"}, "No Smart Markers exist yet.")
            return {"CANCELLED"}
        any_hidden = any(o.hide_viewport or o.hide_get() for o in col.objects)
        set_markers_visibility(any_hidden)
        _tag_all_view3d_redraw(context)
        self.report({"INFO"}, "Smart Markers shown." if any_hidden else "Smart Markers hidden.")
        return {"FINISHED"}


class NURANOVA_OT_clear_markers(bpy.types.Operator):
    bl_idname = "nuranova.clear_markers"
    bl_label = "Clear Markers"
    bl_description = "Remove all Smart Rig Markers from the scene"

    def execute(self, context):
        removed = clear_markers()
        context.scene.nuranova_mocap.guided_marker_prompt = ""
        _tag_all_view3d_redraw(context)
        self.report({"INFO"}, f"Cleared {removed} Smart Markers.")
        return {"FINISHED"}


class NURANOVA_OT_generate_rig_from_markers(bpy.types.Operator):
    bl_idname = "nuranova.generate_rig_from_markers"
    bl_label = "Generate 55-Bone Rig & Bind Mesh"
    bl_description = "Build a precise 55-bone rig directly from your placed Smart Markers, bind Automatic Weights & auto-map"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        meshes = _resolve_target_character_meshes(context, props)
        if not meshes:
            self.report({"ERROR"}, "Pick a Character Mesh or select your character mesh(es) first.")
            return {"CANCELLED"}
        _remember_target_meshes(props, meshes)

        try:
            arm, method = auto_rig_from_markers(
                mesh_objs=meshes,
                include_fingers=props.include_fingers,
                facing_override=props.character_facing,
                hide_markers_after=True,
                context=context,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Smart Rig failed: {exc}")
            return {"CANCELLED"}

        props.target_armature = arm
        props.rig_preset = "NURANOVA"
        props.guided_marker_prompt = f"Rigged '{meshes[0].name}' ({len(arm.data.bones)} bones, {method})!"
        if props.source_armature is not None and props.source_armature != arm:
            mapping = auto_match_bones(props.source_armature, arm, rig_preset="NURANOVA")
            _populate_bone_map_collection(props, mapping)
            self.report(
                {"INFO"},
                f"Generated Smart Rig '{arm.name}' ({len(arm.data.bones)} bones, {method}, {len(meshes)} mesh(es)) & auto-mapped {len(mapping)} bones!",
            )
        else:
            self.report(
                {"INFO"},
                f"Generated Smart Rig '{arm.name}' ({len(arm.data.bones)} bones, {method}, {len(meshes)} mesh(es)) — ready for Live Preview or Retarget!",
            )
        _tag_all_view3d_redraw(context)
        return {"FINISHED"}


# ---------------- Studio Connection, Live Preview & Retarget Operators ----------------

class NURANOVA_OT_connect_studio(bpy.types.Operator):
    bl_idname = "nuranova.connect_studio"
    bl_label = "Check Studio Connection"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        props.studio_status_text = "Connecting..."
        _fetch_json_async(f"{props.studio_url.rstrip('/')}/api/v1/bridge/status", "status")
        return {"FINISHED"}


class NURANOVA_OT_toggle_live_preview(bpy.types.Operator):
    bl_idname = "nuranova.toggle_live_preview"
    bl_label = "Toggle Live Mocap Preview"
    bl_description = "Stream real-time body + 30-finger motion directly from Nuranova Mocap Studio into Blender"

    def execute(self, context):
        global _LAST_LIVE_SEQ
        props = context.scene.nuranova_mocap
        props.live_preview_active = not props.live_preview_active
        if props.live_preview_active:
            _LAST_LIVE_SEQ = -1
            _LIVE_FETCH_IN_FLIGHT.clear()
            props.live_status_text = "Connecting to Studio live stream..."
            self.report({"INFO"}, "Started Live Mocap Preview in Blender.")
        else:
            props.live_status_text = "Live preview stopped."
            self.report({"INFO"}, "Stopped Live Mocap Preview.")
        return {"FINISHED"}


class NURANOVA_OT_create_humanoid_rig(bpy.types.Operator):
    bl_idname = "nuranova.create_humanoid_rig"
    bl_label = "Create Humanoid Rig (Body + Fingers)"
    bl_description = "Spawn a standalone 55-bone Nuranova Humanoid Rig (25 body + 30 fingers)"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        arm = create_nuranova_humanoid_rig(
            name="Nuranova_Humanoid_Rig",
            height_m=props.auto_rig_height_m,
            include_fingers=props.include_fingers,
            context=context,
        )
        props.target_armature = arm
        if props.source_armature is not None and props.source_armature != arm:
            mapping = auto_match_bones(props.source_armature, arm, rig_preset=props.rig_preset)
            _populate_bone_map_collection(props, mapping)
        self.report({"INFO"}, f"Created '{arm.name}' ({len(arm.data.bones)} bones) and assigned as Target Character Rig.")
        return {"FINISHED"}


class NURANOVA_OT_auto_rig_character(bpy.types.Operator):
    bl_idname = "nuranova.auto_rig_character"
    bl_label = "Quick Box Auto-Rig (Realistic)"
    bl_description = "Quick bounding-box Auto-Rig without markers (use Smart Markers above for stylized/custom characters)"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        meshes = _resolve_target_character_meshes(context, props)
        if not meshes:
            self.report({"ERROR"}, "Pick a Character Mesh above or select a Mesh object in the 3D Viewport first.")
            return {"CANCELLED"}
        _remember_target_meshes(props, meshes)

        try:
            arm, method = auto_rig_character_mesh(
                meshes,
                include_fingers=props.include_fingers,
                context=context,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Auto-Rig failed: {exc}")
            return {"CANCELLED"}

        props.target_armature = arm
        props.rig_preset = "NURANOVA"
        if props.source_armature is not None and props.source_armature != arm:
            mapping = auto_match_bones(props.source_armature, arm, rig_preset="NURANOVA")
            _populate_bone_map_collection(props, mapping)
            self.report(
                {"INFO"},
                f"Auto-Rigged '{meshes[0].name}' ({len(arm.data.bones)} bones, {method}) & auto-mapped {len(mapping)} bones!",
            )
        else:
            self.report(
                {"INFO"},
                f"Auto-Rigged '{meshes[0].name}' ({len(arm.data.bones)} bones, {method}) — ready for Live Preview or Retarget!",
            )
        return {"FINISHED"}


class NURANOVA_OT_pull_take(bpy.types.Operator):
    bl_idname = "nuranova.pull_take"
    bl_label = "Pull Take from Studio"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        if not props.selected_take_id:
            self.report({"WARNING"}, "Enter or discover a Take ID first.")
            return {"CANCELLED"}
        props.studio_status_text = f"Pulling take '{props.selected_take_id}'..."
        _fetch_json_async(
            f"{props.studio_url.rstrip('/')}/api/v1/bridge/takes/{props.selected_take_id}",
            "import_take",
        )
        return {"FINISHED"}


class NURANOVA_OT_import_offline(bpy.types.Operator):
    bl_idname = "nuranova.import_offline"
    bl_label = "Import Offline Motion (.nmocap.json / .bvh)"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.motion_filepath)
        if not fp or not Path(fp).exists():
            self.report({"ERROR"}, f"Motion file not found: {fp}")
            return {"CANCELLED"}

        if fp.lower().endswith(".bvh"):
            bpy.ops.import_anim.bvh(filepath=fp)
            if context.active_object and context.active_object.type == "ARMATURE":
                props.source_armature = context.active_object
            self.report({"INFO"}, f"Imported BVH: {Path(fp).name}")
            return {"FINISHED"}

        payload = load_nmocap_json(fp)
        _LAST_PAYLOAD_CACHE["latest"] = payload
        arm = create_source_armature_from_payload(
            payload,
            in_place=props.in_place,
            push_nla=props.push_to_nla,
            include_fingers=props.include_fingers,
            context=context,
        )
        props.source_armature = arm
        if props.target_armature is not None and props.target_armature != arm:
            mapping = auto_match_bones(arm, props.target_armature, rig_preset=props.rig_preset)
            _populate_bone_map_collection(props, mapping)
        self.report({"INFO"}, f"Imported take '{payload.get('take_id')}' ({payload.get('num_frames')} frames, {len(arm.data.bones)} bones)")
        return {"FINISHED"}


class NURANOVA_OT_auto_map(bpy.types.Operator):
    bl_idname = "nuranova.auto_map_bones"
    bl_label = "Auto-Match Rig Bones (Body + Fingers)"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        if not props.source_armature or not props.target_armature:
            self.report({"ERROR"}, "Select both Source Rig and Target Character Rig first.")
            return {"CANCELLED"}
        mapping = auto_match_bones(props.source_armature, props.target_armature, rig_preset=props.rig_preset)
        _populate_bone_map_collection(props, mapping)
        self.report({"INFO"}, f"Matched {len(mapping)} bones automatically ({props.rig_preset}).")
        return {"FINISHED"}


class NURANOVA_OT_save_mapping(bpy.types.Operator):
    bl_idname = "nuranova.save_mapping"
    bl_label = "Save Mapping Profile"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.mapping_filepath)
        if not fp:
            self.report({"ERROR"}, "Specify a Mapping Profile file path first.")
            return {"CANCELLED"}
        mapping = {it.source_bone: it.target_bone for it in props.bone_map_items if it.source_bone and it.target_bone}
        rig_name = props.target_armature.name if props.target_armature else ""
        save_mapping_profile(mapping, fp, rig_name=rig_name)
        self.report({"INFO"}, f"Saved mapping profile ({len(mapping)} bones) to {fp}")
        return {"FINISHED"}


class NURANOVA_OT_load_mapping(bpy.types.Operator):
    bl_idname = "nuranova.load_mapping"
    bl_label = "Load Mapping Profile"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.mapping_filepath)
        if not fp or not Path(fp).exists():
            self.report({"ERROR"}, f"Mapping file not found: {fp}")
            return {"CANCELLED"}
        mapping = load_mapping_profile(fp)
        _populate_bone_map_collection(props, mapping)
        self.report({"INFO"}, f"Loaded {len(mapping)} bone mappings.")
        return {"FINISHED"}


class NURANOVA_OT_retarget_bake(bpy.types.Operator):
    bl_idname = "nuranova.retarget_bake"
    bl_label = "Retarget & Bake Editable Keyframes"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        if not props.source_armature or not props.target_armature:
            self.report({"ERROR"}, "Select both Source Rig and Target Character Rig.")
            return {"CANCELLED"}
        mapping = {it.source_bone: it.target_bone for it in props.bone_map_items if it.source_bone and it.target_bone}
        if not mapping:
            mapping = auto_match_bones(props.source_armature, props.target_armature, rig_preset=props.rig_preset)
            _populate_bone_map_collection(props, mapping)
        act = retarget_and_bake(
            source_arm=props.source_armature,
            target_arm=props.target_armature,
            bone_map=mapping,
            frame_start=context.scene.frame_start,
            frame_end=context.scene.frame_end,
            scale_compensation=props.scale_compensation,
            extract_root_motion=props.extract_root_motion,
            in_place=props.in_place,
            ground_to_floor=props.ground_to_floor,
            push_nla=props.push_to_nla,
            context=context,
        )
        payload = _LAST_PAYLOAD_CACHE.get("latest")
        if payload:
            if props.target_face_mesh:
                apply_facial_blendshapes_to_mesh(props.target_face_mesh, payload, gain=props.face_gain)
            if props.include_fingers and not _mapping_contains_fingers(mapping):
                apply_hand_rotations_to_armature(props.target_armature, payload, include_wrist=False)
        self.report({"INFO"}, f"Baked editable Action '{act.name}' ({len(mapping)} bones).")
        return {"FINISHED"}


class NURANOVA_OT_export_fbx(bpy.types.Operator):
    bl_idname = "nuranova.export_fbx"
    bl_label = "Export FBX"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.export_filepath)
        if not fp.lower().endswith(".fbx"):
            fp = str(Path(fp).with_suffix(".fbx"))
        from .character_export import export_character
        try:
            report = export_character(props.target_armature or props.source_armature, fp, variant=props.export_variant)
        except (ValueError, RuntimeError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Verified FBX: {report['samples']} poses · {report['max_joint_error_m'] * 1000:.2f}mm")
        return {"FINISHED"}


class NURANOVA_OT_export_glb(bpy.types.Operator):
    bl_idname = "nuranova.export_glb"
    bl_label = "Export animated GLB"

    def execute(self, context):
        from .character_export import export_character
        props = context.scene.nuranova_mocap
        fp = str(Path(bpy.path.abspath(props.export_filepath)).with_suffix(".glb"))
        try:
            report = export_character(props.target_armature or props.source_armature, fp, variant=props.export_variant)
        except (ValueError, RuntimeError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Verified GLB: {report['samples']} poses · {report['max_joint_error_m'] * 1000:.2f}mm")
        return {"FINISHED"}


class NURANOVA_OT_export_blend(bpy.types.Operator):
    bl_idname = "nuranova.export_blend"
    bl_label = "Save editable BLEND copy"

    def execute(self, context):
        path = Path(bpy.path.abspath(context.scene.nuranova_mocap.export_filepath)).with_suffix(".blend").resolve()
        if bpy.data.filepath and path == Path(bpy.data.filepath).resolve():
            self.report({"ERROR"}, "Choose another export filename to preserve the open project.")
            return {"CANCELLED"}
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True)
        self.report({"INFO"}, f"Saved editable scene copy: {path}")
        return {"FINISHED"}


class NURANOVA_OT_export_bvh(bpy.types.Operator):
    bl_idname = "nuranova.export_bvh"
    bl_label = "Export BVH"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.export_filepath)
        if not fp.lower().endswith(".bvh"):
            fp = str(Path(fp).with_suffix(".bvh"))
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        arm = props.target_armature or props.source_armature
        if arm is None:
            self.report({"ERROR"}, "Select an armature to export BVH.")
            return {"CANCELLED"}
        bpy.ops.object.select_all(action="DESELECT")
        arm.select_set(True)
        context.view_layer.objects.active = arm
        bpy.ops.export_anim.bvh(
            filepath=fp,
            frame_start=context.scene.frame_start,
            frame_end=context.scene.frame_end,
            root_transform_only=False,
        )
        self.report({"INFO"}, f"Exported BVH to {fp}")
        return {"FINISHED"}


class NURANOVA_UL_bone_map(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "source_bone", text="", emboss=False)
        row.label(text="->")
        row.prop(item, "target_bone", text="")


class NURANOVA_PT_main_panel(bpy.types.Panel):
    bl_label = "Nuranova Mocap Bridge"
    bl_idname = "NURANOVA_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Nuranova Mocap"

    def draw(self, context):
        layout = self.layout
        props = context.scene.nuranova_mocap

        # 1. Studio Connection & Real-Time Live Preview
        box = layout.box()
        box.label(text="1. Studio Connection & Live Preview", icon="LINKED")
        box.prop(props, "studio_url")
        row = box.row(align=True)
        row.operator("nuranova.connect_studio", icon="FILE_REFRESH")
        row.operator("nuranova.pull_take", icon="IMPORT")
        box.prop(props, "selected_take_id")
        box.label(text=props.studio_status_text)

        live_row = box.row()
        live_row.scale_y = 1.25
        live_row.operator(
            "nuranova.toggle_live_preview",
            text="Stop Live Preview" if props.live_preview_active else "Start Live Preview in Blender",
            icon="PAUSE" if props.live_preview_active else "PLAY",
            depress=props.live_preview_active,
        )
        box.label(text=props.live_status_text)

        # 2. Smart Marker Auto-Rig (Works on Stylized, Anime, Cartoon & Realistic Characters!)
        box_rig = layout.box()
        box_rig.label(text="2. Smart Marker Auto-Rig (Body + 30 Fingers)", icon="OUTLINER_OB_ARMATURE")
        box_rig.prop(props, "rig_preset")
        box_rig.prop(props, "character_facing")
        row_opts = box_rig.row(align=True)
        row_opts.prop(props, "include_fingers")
        row_opts.prop(props, "mirror_markers", text="Mirror L/R")
        box_rig.prop(props, "target_character_mesh")

        placed, total = count_placed_body_landmarks()
        status_col = box_rig.column(align=True)
        if placed >= total:
            status_col.label(text=f"Smart Markers: {placed}/{total} Ready! (Drag any marker with G)", icon="CHECKMARK")
        elif placed > 0:
            status_col.label(text=f"Smart Markers: {placed}/{total} placed", icon="PIVOT_CURSOR")
        else:
            status_col.label(text="Step A: Select mesh(es) -> Place 12 Guided Markers", icon="INFO")

        if props.guided_marker_prompt:
            status_col.label(text=props.guided_marker_prompt)

        row_guide = box_rig.row()
        row_guide.scale_y = 1.25
        row_guide.operator("nuranova.guided_place_markers", icon="EYEDROPPER")

        row_mk1 = box_rig.row(align=True)
        row_mk1.operator("nuranova.spawn_preset_markers", text="Place Preset Markers", icon="EMPTY_AXIS")
        row_mk1.operator("nuranova.spawn_finger_markers", text="Finger Markers", icon="VIEW_PAN")

        row_mk2 = box_rig.row(align=True)
        row_mk2.operator("nuranova.mirror_markers", icon="MOD_MIRROR")
        row_mk2.operator("nuranova.toggle_markers", text="Show/Hide", icon="HIDE_OFF")
        row_mk2.operator("nuranova.clear_markers", text="Clear", icon="TRASH")

        row_gen = box_rig.row()
        row_gen.scale_y = 1.3
        row_gen.enabled = placed > 0
        row_gen.operator("nuranova.generate_rig_from_markers", icon="MOD_ARMATURE")

        # Quick utilities (for standalone rig or quick realistic box fit)
        row_util = box_rig.row(align=True)
        row_util.operator("nuranova.auto_rig_character", text="Quick Box Rig", icon="MESH_CUBE")
        row_util.operator("nuranova.create_humanoid_rig", text="New 55-Bone Rig", icon="ARMATURE_DATA")

        # 3. Offline Motion Import
        box2 = layout.box()
        box2.label(text="3. Offline Motion Import", icon="FILE_FOLDER")
        box2.prop(props, "motion_filepath")
        box2.operator("nuranova.import_offline", icon="ARMATURE_DATA")

        # 4. Auto-Retarget & Bake
        box3 = layout.box()
        box3.label(text="4. Auto-Retarget & Bake (Body + Fingers)", icon="POSE_HLT")
        box3.prop(props, "source_armature")
        box3.prop(props, "target_armature")
        box3.prop(props, "target_face_mesh")
        box3.operator("nuranova.auto_map_bones", icon="AUTO")
        box3.template_list("NURANOVA_UL_bone_map", "", props, "bone_map_items", props, "bone_map_index", rows=5)
        box3.prop(props, "mapping_filepath")
        row_m = box3.row(align=True)
        row_m.operator("nuranova.save_mapping", icon="FILE_TICK")
        row_m.operator("nuranova.load_mapping", icon="FILE_FOLDER")
        box3.prop(props, "scale_compensation")
        box3.prop(props, "extract_root_motion")
        box3.prop(props, "ground_to_floor")
        box3.prop(props, "in_place")
        box3.prop(props, "push_to_nla")
        box3.prop(props, "face_gain")
        box3.operator("nuranova.retarget_bake", icon="ACTION")
        draw_quality_panel(layout, props)

        # 5. Export
        box4 = layout.box()
        box4.label(text="5. Animation Export", icon="EXPORT")
        box4.prop(props, "export_filepath")
        box4.prop(props, "export_variant")
        row_e = box4.row(align=True)
        row_e.operator("nuranova.export_fbx", icon="EXPORT")
        row_e.operator("nuranova.export_glb", icon="EXPORT")
        row_e.operator("nuranova.export_bvh", icon="EXPORT")
        box4.operator("nuranova.export_blend", icon="FILE_BLEND")


_CLASSES = (
    NuranovaBoneMapItem,
    NuranovaMocapProperties,
    NURANOVA_OT_guided_place_markers,
    NURANOVA_OT_spawn_preset_markers,
    NURANOVA_OT_spawn_finger_markers,
    NURANOVA_OT_mirror_markers,
    NURANOVA_OT_toggle_markers,
    NURANOVA_OT_clear_markers,
    NURANOVA_OT_generate_rig_from_markers,
    NURANOVA_OT_connect_studio,
    NURANOVA_OT_toggle_live_preview,
    NURANOVA_OT_create_humanoid_rig,
    NURANOVA_OT_auto_rig_character,
    NURANOVA_OT_pull_take,
    NURANOVA_OT_import_offline,
    NURANOVA_OT_auto_map,
    NURANOVA_OT_save_mapping,
    NURANOVA_OT_load_mapping,
    NURANOVA_OT_retarget_bake,
    NURANOVA_OT_export_fbx,
    NURANOVA_OT_export_glb,
    NURANOVA_OT_export_blend,
    *QUALITY_CLASSES,
    NURANOVA_OT_export_bvh,
    NURANOVA_UL_bone_map,
    NURANOVA_PT_main_panel,
)


def register() -> None:
    global _PREVIEW_3D_HANDLER
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.nuranova_mocap = PointerProperty(type=NuranovaMocapProperties)
    if not bpy.app.timers.is_registered(_bridge_main_thread_timer):
        bpy.app.timers.register(_bridge_main_thread_timer, persistent=True)
    if not bpy.app.background and _PREVIEW_3D_HANDLER is None:
        _PREVIEW_3D_HANDLER = bpy.types.SpaceView3D.draw_handler_add(
            _draw_smart_marker_skeleton_3d, (), "WINDOW", "POST_VIEW"
        )


def unregister() -> None:
    global _PREVIEW_3D_HANDLER
    if _PREVIEW_3D_HANDLER is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_PREVIEW_3D_HANDLER, "WINDOW")
        except Exception:
            pass
        _PREVIEW_3D_HANDLER = None
    if bpy.app.timers.is_registered(_bridge_main_thread_timer):
        bpy.app.timers.unregister(_bridge_main_thread_timer)
    if hasattr(bpy.types.Scene, "nuranova_mocap"):
        del bpy.types.Scene.nuranova_mocap
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
