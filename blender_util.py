import bpy

def ensure_object_mode():
    """Headless/后台安全：尽量切到 Object 模式 """
    try:
        if bpy.context.mode != 'OBJECT':
            
    except Exception:



def deselect_all():
    """尽量不依赖 ops；不行再 fallback 到 select_all(DESELECT)。"""
    try:
        for o in list(bpy.context.selected_objects):
            o.select_set(False)
    except Exception:
        if bpy.ops.object.select_all.poll():
            bpy.ops.object.select_all(action='DESELECT')

def select_only(objs, active=None):
    """强制 selection/active，减少后续 ops 的 context 错误。"""
    ensure_object_mode()
    deselect_all()
    for o in objs:
        if o and o.name in bpy.data.objects:
            o.select_set(True)
    if active and active.name in bpy.data.objects:
        bpy.context.view_layer.objects.active = active

def delete_objects(objs):
    """删除一组对象：优先 ops.delete，失败则 data.remove 兜底。"""
    objs = [o for o in (objs or []) if o and o.name in bpy.data.objects]
    if not objs:
        return

    ensure_object_mode()
    select_only(objs, active=objs[0])

    if bpy.ops.object.delete.poll():
        bpy.ops.object.delete()
        return

    # fallback: 直接 remove（不依赖上下文）
    for o in objs:
        if o and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)

def join_into_active(root, others):
    """
    将 others join 到 root（root 保留）。
    返回 root（若 join poll 失败则原样返回）。
    """
    if root is None or root.name not in bpy.data.objects:
        return root
    others = [o for o in (others or []) if o and o.name in bpy.data.objects and o != root]
    if not others:
        return root

    ensure_object_mode()
    select_only([root] + others, active=root)

    if bpy.ops.object.join.poll():
        bpy.ops.object.join()
    return root

def apply_modifier(obj, modifier_name):
    """对 obj 应用指定 modifier（依赖 active object，统一在这里处理）。"""
    if obj is None or obj.name not in bpy.data.objects:
        return False

    ensure_object_mode()
    select_only([obj], active=obj)

    if modifier_name not in obj.modifiers:
        return False

    if bpy.ops.object.modifier_apply.poll():
        bpy.ops.object.modifier_apply(modifier=modifier_name)
        return True

    return False

def convert_object_to_mesh_object(obj, depsgraph):
    """
    无 UI / headless 安全的“转 Mesh”（不使用 bpy.ops.object.convert）：
    - evaluated object -> mesh datablock
    - 新建 MESH object 替换原对象（保留 transform / collection / parent / children）
    返回：替换后的 mesh object（或原对象如果无需转换/无法转换）
    """
    if obj is None or obj.name not in bpy.data.objects:
        return obj

    if obj.type == 'MESH':
        return obj

    if obj.type not in {'CURVE', 'SURFACE', 'META', 'FONT', 'GPENCIL'}:
        return obj

    eval_obj = obj.evaluated_get(depsgraph)

    try:
        mesh = bpy.data.meshes.new_from_object(
            eval_obj,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
    except TypeError:
        mesh = bpy.data.meshes.new_from_object(eval_obj)

    if mesh is None:
        return obj

    mesh.name = f"{obj.name}_MESH"
    new_obj = bpy.data.objects.new(obj.name, mesh)
    new_obj.matrix_world = obj.matrix_world.copy()

    cols = list(obj.users_collection)
    if cols:
        for c in cols:
            c.objects.link(new_obj)
    else:
        bpy.context.scene.collection.objects.link(new_obj)

    new_obj.parent = obj.parent
    new_obj.parent_type = obj.parent_type
    new_obj.parent_bone = obj.parent_bone
    new_obj.matrix_parent_inverse = obj.matrix_parent_inverse.copy()

    for child in list(obj.children):
        child_mw = child.matrix_world.copy()
        child.parent = new_obj
        child.matrix_parent_inverse = new_obj.matrix_world.inverted() @ child_mw

    bpy.data.objects.remove(obj, do_unlink=True)
    return new_obj