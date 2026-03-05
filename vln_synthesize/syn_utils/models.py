"""Data models for solve_state.json parsing."""

import re
from dataclasses import dataclass, field
from typing import ClassVar, Mapping, cast

import isaacsim.core.utils.bounds as bounds_utils # type: ignore
import isaacsim.core.utils.prims as prims_utils   # type: ignore
from numpy import ndarray   # type: ignore
from pxr import Sdf, Usd

MESH_TYPES = {
    "room", "object", "cutter"
}
ROOM_TYPES = {
    "kitchen", "bedroom", "living-room", "closet", "hallway", "bathroom",
    "garage", "balcony", "dining-room", "utility", "staircase-room",
    "warehouse", "office", "meeting-room", "open-office", "break-room",
    "restroom", "factory-office"
}
BUILDING_TYPES = {
    "root", "new", "room-node", "ground", "second-floor", "third-floor",
    "exterior", "stairccase", "visited", "room-contour"
}
OBJECT_TYPES = {
    "furniture", "FloorMat", "wall-decoration", "handheld-item"
}
FURNITURE_FUNCTIONS = {
    "storage", "seating", "lounge-seating", "table", "bathing", "side-table",
    "watchable", "desk", "bed", "sink", "ceiling-light", "lighting",
    "kitchen-counter", "kitchen-appliance"
}
SMALL_OBJECT_FUNCTIONS = {
    "table-display-item",
    "office-shelf-item", "kitchen-counter-item", "food-pantry", "bathroom-item",
    "shelf-trinket", "dishware", "cookware", "utensils", "cloth-drape"
}
OBJECT_ACCESS_TYPE = {
    "access-top", "access-front", "access-any-side", "access-all-sides",
}
OBJECT_ACCESS_METHOD = {
    "access-stand-near", "access-open-door", "access-with-hand",
}
SPECIAL_CASE_OBJECTS = {
    "chair", "window", "open", "entrance", "door",
}
BEHAVIOR_FLAGS = {
    "real-placeholder", "oversize-placeholder", "asset-as-placeholder",
    "asset-placeholder-for-children", "placeholder-bbox", "single-generator",
    "no-rotation", "no-collision", "no-children"
}
SUBPART = {
    "support", "interior", "visible", 
    "bottom", "top", "side", "back", "front", 
    "ceiling", "wall", 
    # "staircase-wall" todo from infinigen
}

TAG_PATTERN: re.Pattern = re.compile(r"^(-?)(\w+)\((.+)\)$")

Matrix3x3 = list[list[float]]
Vector3 = list[float]

StateId = str
Tag = str
TagCollection = list[Tag]
Tags = dict[str, TagCollection]

def parseTag(raw_tags: list[str]) -> Tags:
    tags: dict[str, list[str]] = {}
    for tag_str in raw_tags:
        m = TAG_PATTERN.match(tag_str.strip())
        if m:
            if m.group(1):
                tags.setdefault(m.group(2), []).append(
                    f"-{m.group(3)}" 
                )
            else:
                tags.setdefault(m.group(2), []).append(
                    m.group(3) 
                )
    return tags

# Relationship 定义
@dataclass
class Relation:
    relation_type: str

    @classmethod
    def from_dict(cls, rel_value: dict) -> "Relation":
        relation_type = rel_value.get("relation_type", "")
        match relation_type:
            case "AnyRelation" | "IdentityCompareRelation" | "CutFrom" | "SharedEdge" | "Traverse":
                return Relation(relation_type=relation_type)
            case "RoomNeighbour":
                return RoomNeighbour(
                    relation_type=relation_type,
                    connector_typers=rel_value.get("connector_types", []),
                )
            case "GeometryRelation" | "Touching" | "SupportedBy" | "CoPlanar":
                return GeometryRelation(
                    relation_type=relation_type,
                    child_tags=parseTag(rel_value.get("child_tags", [])),
                    parent_tags=parseTag(rel_value.get("parent_tags", [])),
                )
            case "StableAgainst":
                return StableAgainst(
                    relation_type = relation_type,
                    child_tags = parseTag(rel_value.get("child_tags", [])),
                    parent_tags = parseTag(rel_value.get("parent_tags", [])),
                    margin = float(rel_value.get("margin", 0)),
                    check_z = bool(rel_value.get("check_z", False)),
                    rev_normal = bool(rel_value.get("rev_normal", False)),
                )
            case _:
                return Relation(relation_type=relation_type)

CONNECTOR_TYPES = {"door", "open", "wall"}
@dataclass
class RoomNeighbour(Relation):
    connector_typers: list[str]

@dataclass
class GeometryRelation(Relation):
    child_tags: dict[str, list[str]]
    parent_tags: dict[str, list[str]]

@dataclass
class StableAgainst(GeometryRelation):
    margin: float = 0
    check_z: bool = False
    rev_normal: bool = False


# Relationship 包装
@dataclass
class RelationState:
    ''' 这是一个 Relation 的包装类，有效字段如下
    child_plane_idx
    parent_plane_idx
    target_name
    relation
    '''
    child_plane_idx: int | None
    parent_plane_idx: int | None
    target_name: StateId
    value: str | None
    relation: Relation
    
    @classmethod
    def from_dict(cls, rel_item: dict) -> "RelationState":
        return cls(
            child_plane_idx = rel_item.get("child_plane_idx"),
            parent_plane_idx = rel_item.get("parent_plane_idx"),
            target_name = rel_item.get("target_name", ""),
            value = rel_item.get("value"),
            relation = Relation.from_dict(rel_item.get("relation", {}))
        )
    
NavigationPoint = list[float]
NavigationPointCollection = list[NavigationPoint]

@dataclass
class Prims:
    prims: dict[Sdf.Path, Usd.Prim]

    # post init variables
    bbox: list[float] | None = field(init=False)
    obb: tuple[ndarray, ndarray, ndarray] | None = field(init=False)

    def __post_init__(self):
        if len(self.prims) == 0:
            self.bbox = None
            self.obb = None
            return
        cache = bounds_utils.create_bbox_cache()
        self.bbox = bounds_utils.compute_combined_aabb(
            cache, 
            prim_paths=[p for p in self.prims.keys()]
        )
        self.obb = bounds_utils.compute_obb(
            cache, 
            list(self.prims.keys())[0]
        )
        
    @property
    def first_prim(self) -> Usd.Prim | None:
        if not self.prims:
            return None
        return list(self.prims.values())[0]


STATE_TYPES : dict[str, type["State"]] = {}

def state(state_type: str):
    def inner(cls):
        STATE_TYPES[state_type] = cls
        cls.state_type = state_type
        return cls
    return inner


# State 定义, 基类
@state("base")
@dataclass
class State:
    id: StateId
    obj_name: str
    generator: str | None
    active: bool
    tags: Tags
    relations: list[RelationState]
    dof_matrix_translation: Matrix3x3 | None
    dof_rotation_axis: Vector3 | None
    state_type: ClassVar[str]  # ClassVar 表示此变量属于类，dataclass 将不在 init 中实现它 [PEP 557]. 否则装饰器对state_type的设置不起作用

    prims: Prims = field(init=False)
    navigation_points : NavigationPointCollection = field(default_factory=list)

    # post-init 计算的字段
    prim_name: str = field(init=False, default="")
    def __post_init__(self):
        if self.obj_name.endswith(".meshed"):
            base = self.obj_name[:-7]
        else:
            base = self.obj_name
        self.prim_name = base.replace("-", "_").replace("/", "_").replace(".", "_").replace("(", "_").replace(")", "_")

    @property
    def semantic_tags(self) -> list[str]:
        return self.tags.get("Semantics", [])

    @classmethod
    def from_dict(cls, obj_key, obj_value: dict) -> "State":
        tags = parseTag(obj_value.get("tags", []))
        semantic_tags = tags.get("Semantics", [])
        
        klass = next((sval for skey, sval in STATE_TYPES.items() if skey in semantic_tags), State)

        return klass(
            id = obj_key,
            obj_name = obj_value.get("obj", ""),
            generator = obj_value.get("generator"),
            active = bool(obj_value.get("active")),
            tags = tags,
            relations = [RelationState.from_dict(r) for r in obj_value.get("relations", [])],
            dof_matrix_translation = obj_value.get("dof_matrix_translation"),
            dof_rotation_axis = obj_value.get("dof_rotation_axis"),
        )

    def computePrims(self, stage: Usd.Stage):
        prim_path = Sdf.Path(f"/World/{self.prim_name}")
        self.prims = Prims({prim_path: stage.GetPrimAtPath(prim_path)} 
                               if prims_utils.is_prim_path_valid(prim_path) 
                               else {})

@state("room")
@dataclass
class RoomState(State):
    navigation_access : set[StateId] = field(default_factory=set)
    objects : set[StateId] = field(default_factory=set)

    @property
    def room_type(self) -> str | None:
        return next((v for v in self.semantic_tags if v in ROOM_TYPES), None)
    
    def computePrims(self, stage: Usd.Stage):
        prims = {}
        for suffix in {"_floor", "_ceiling", "_wall", "_exterior", ""}:
            prim_path = Sdf.Path(f"/World/{self.prim_name}{suffix}")
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                prims[prim_path] = prim
        self.prims = Prims(prims)

# ObjectState 特指 semantic tag 为object的对象
@state("object")
@dataclass
class ObjectState(State):
    room : RoomState | None = None

    @property
    def object_type(self) -> str | None:
        return next((v for v in self.semantic_tags if v in OBJECT_TYPES), None)
    
    @property
    def access_type(self) -> str | None:
        return next((v for v in self.semantic_tags if v in OBJECT_ACCESS_TYPE), None)
    
    @property
    def access_method(self) -> str | None:
        return next((v for v in self.semantic_tags if v in OBJECT_ACCESS_METHOD), None)
    
    @property
    def function_tags(self) -> str | None:
        return next((v for v in self.semantic_tags if v in FURNITURE_FUNCTIONS or v in SMALL_OBJECT_FUNCTIONS), None)
    
    @property
    def parent_id(self) -> str | None:
        for rel in self.relations:
            if rel.relation.relation_type == "StableAgainst":
                return rel.target_name
        return None
    

    def computeRoom(self, states: "States"):
        rooms = states.get("room", {})
        rooms = cast(dict[str, RoomState], rooms)
        objects = states.get("object", {})
        objects = cast(dict[str, ObjectState], objects)

        parent_id = self.parent_id
        while parent_id:
            if (room := rooms.get(parent_id)):
                self.room = room
                room.objects.add(self.id)
                break

            if (obj := objects.get(parent_id)):
                parent_id = obj.parent_id


@state("cutter")
@dataclass
class CutterState(State):

    @property
    def cutter_type(self) -> str | None:
        return next((v for v in self.semantic_tags if v in SPECIAL_CASE_OBJECTS), None)
    

States = dict[str, dict[str, State]]
ReadonlyStates = Mapping[str, Mapping[str, State]]
