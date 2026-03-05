import carb
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


def createPhysicsScene(stage: Usd.Stage):
    """Create a PhysicsScene if none exists."""
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            carb.log_info(f"Physics scene already exists at {prim.GetPath()}")
            return prim.GetPath()
    stage.DefinePrim("/PhysicsScene", "PhysicsScene")
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    carb.log_info("Created /PhysicsScene")
    return "/PhysicsScene"

def setPhysicsScene(stage: Usd.Stage):
    """Configure the PhysicsScene with PhysX parameters."""
    scene_prim = stage.GetPrimAtPath(createPhysicsScene(stage))
    physx_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_api.CreateEnableCCDAttr(True)
    physx_api.CreateEnableGPUDynamicsAttr(False)


def setAllMeshCollision(stage: Usd.Stage):
    """Add collision APIs to all mesh prims for raycasting."""
    count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim)
                count += 1
    carb.log_info(f"Added collision to {count} meshes")
    return count
