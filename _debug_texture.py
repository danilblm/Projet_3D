"""Quick diagnostic: check if trimesh preserves UVs and textures."""
import trimesh
import numpy as np

MODEL = "models/bmw_m4_widebody__www.vecarz.com.glb"

print("=== Test 1: force='mesh' ===")
m1 = trimesh.load(MODEL, force="mesh")
print(f"Type: {type(m1).__name__}")
print(f"Vertices: {len(m1.vertices)}")
print(f"Visual type: {type(m1.visual).__name__}")
if hasattr(m1.visual, "uv") and m1.visual.uv is not None:
    uv = np.array(m1.visual.uv)
    print(f"UV shape: {uv.shape}")
    print(f"UV range: [{uv.min():.4f}, {uv.max():.4f}]")
    print(f"UV all zeros? {np.allclose(uv, 0)}")
else:
    print("NO UV data!")

if hasattr(m1.visual, "material"):
    mat = m1.visual.material
    print(f"Material type: {type(mat).__name__}")
    for attr in ["image", "baseColorTexture", "baseColorFactor"]:
        val = getattr(mat, attr, None)
        if val is not None:
            print(f"  {attr}: {type(val).__name__} {getattr(val, 'size', '')}")
else:
    print("No material")

print()
print("=== Test 2: Scene load ===")
scene = trimesh.load(MODEL)
print(f"Type: {type(scene).__name__}")
if isinstance(scene, trimesh.Scene):
    for name, geom in scene.geometry.items():
        vt = type(geom.visual).__name__
        print(f"  Mesh '{name}': {len(geom.vertices)} verts, visual={vt}")
        if hasattr(geom.visual, "uv") and geom.visual.uv is not None:
            uv = geom.visual.uv
            print(f"    UV: {uv.shape}, range=[{uv.min():.3f}, {uv.max():.3f}]")
        if hasattr(geom.visual, "material"):
            mat = geom.visual.material
            for attr in ["image", "baseColorTexture", "baseColorFactor"]:
                val = getattr(mat, attr, None)
                if val is not None:
                    print(f"    {attr}: {type(val).__name__} size={getattr(val, 'size', 'N/A')}")
    print()
    merged = scene.dump(concatenate=True)
    print(f"Merged: {len(merged.vertices)} verts, visual={type(merged.visual).__name__}")
    if hasattr(merged.visual, "uv") and merged.visual.uv is not None:
        uv = merged.visual.uv
        print(f"  Merged UV: {uv.shape}, range=[{uv.min():.3f}, {uv.max():.3f}]")
    else:
        print("  Merged: NO UV!")
