"""
opengl_renderer.py — Off-screen OpenGL 3D renderer via moderngl.

Features:
  • Bakes textures + materials into per-vertex colours (robust, no UV issues)
  • swap_model() for instant model switching (GPU buffer swap only)
  • preload_mesh() static helper to do heavy work upfront
  • Blinn-Phong lighting + ground shadow
"""

from __future__ import annotations

import os
import sys
import numpy as np
import trimesh
import moderngl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─────────────────────────────────────────────────────
# GLSL — main model
# ─────────────────────────────────────────────────────

VERTEX_SHADER = """
#version 330 core

uniform mat4 u_mvp;
uniform mat4 u_mv;
uniform mat3 u_normal_mat;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_normal_eye;
out vec3 v_pos_eye;
out vec3 v_color;

void main() {
    vec4 pos_eye  = u_mv * vec4(in_position, 1.0);
    v_pos_eye     = pos_eye.xyz;
    v_normal_eye  = normalize(u_normal_mat * in_normal);
    v_color       = in_color;
    gl_Position   = u_mvp * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal_eye;
in vec3 v_pos_eye;
in vec3 v_color;

out vec4 frag_color;

const vec3  LIGHT_DIR = normalize(vec3(0.3, 0.8, 0.6));
const float AMBIENT   = 0.30;
const float DIFFUSE   = 0.60;
const float SPECULAR  = 0.25;
const float SHININESS = 32.0;

void main() {
    vec3 N = normalize(v_normal_eye);
    vec3 V = normalize(-v_pos_eye);
    if (dot(N, V) < 0.0) N = -N;

    float diff = max(dot(N, LIGHT_DIR), 0.0);
    vec3  H    = normalize(LIGHT_DIR + V);
    float spec = pow(max(dot(N, H), 0.0), SHININESS);

    vec3 color = v_color * (AMBIENT + DIFFUSE * diff)
               + vec3(1.0) * SPECULAR * spec;

    frag_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""

# ─────────────────────────────────────────────────────
# GLSL — shadow
# ─────────────────────────────────────────────────────

SHADOW_VERTEX = """
#version 330 core
uniform mat4 u_mvp;
in vec3 in_position;
out vec2 v_pos;
void main() {
    v_pos = in_position.xz;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

SHADOW_FRAGMENT = """
#version 330 core
in vec2 v_pos;
out vec4 frag_color;
uniform float u_radius;
void main() {
    float d = length(v_pos) / u_radius;
    float alpha = 0.4 * smoothstep(1.0, 0.1, d);
    frag_color = vec4(0.0, 0.0, 0.0, alpha);
}
"""


def _make_shadow_quad(radius):
    r = radius
    v = np.array([[-r, 0, -r], [r, 0, -r], [r, 0, r], [-r, 0, r]],
                 dtype=np.float32)
    i = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return v, i


# ─────────────────────────────────────────────────────
# Mesh data container (CPU-side, preloaded)
# ─────────────────────────────────────────────────────

class MeshData:
    """Holds pre-processed mesh arrays ready for GPU upload."""
    __slots__ = ("vertices", "normals", "colors", "indices", "name")

    def __init__(self, vertices, normals, colors, indices, name=""):
        self.vertices = vertices
        self.normals = normals
        self.colors = colors
        self.indices = indices
        self.name = name


def preload_mesh(path: str) -> MeshData:
    """
    Load and process a 3D model on the CPU.
    This does the heavy work (disk I/O, texture baking, normalisation).
    Call this once per model at startup.
    """
    name = os.path.basename(path)
    print(f"[preload] Loading {name} ...", end=" ", flush=True)

    mesh = trimesh.load(path, force="mesh")

    # Auto-decimate heavy meshes
    max_faces = config.MAX_FACES
    if max_faces > 0 and len(mesh.faces) > max_faces:
        try:
            print(f"(decimating {len(mesh.faces)}->{max_faces} faces) ", end="", flush=True)
            mesh = mesh.simplify_quadric_decimation(max_faces)
        except Exception:
            print("(decimation unavailable, using full mesh) ", end="", flush=True)

    # Bake textures/materials into vertex colours
    if hasattr(mesh.visual, "to_color"):
        mesh.visual = mesh.visual.to_color()

    # Centre the mesh
    mesh.vertices -= mesh.centroid

    # ── Auto-orient: ensure the shortest bounding-box axis points +Y (up) ──
    # For vehicles: length > width > height, so height = shortest axis.
    bbox = mesh.bounding_box.extents  # (sx, sy, sz)
    axes_lengths = list(bbox)
    shortest_axis = int(np.argmin(axes_lengths))
    if shortest_axis == 0:      # X is shortest → rotate 90° around Z so X→Y
        R = trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1])
        mesh.apply_transform(R)
    elif shortest_axis == 2:    # Z is shortest → rotate -90° around X so Z→Y
        R = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        mesh.apply_transform(R)
    # If shortest_axis == 1 (Y), it's already correct

    # Re-centre after rotation
    mesh.vertices -= mesh.centroid

    # Scale to marker size
    radius = mesh.bounding_sphere.primitive.radius
    s = config.MARKER_LENGTH_M / radius
    mesh.apply_scale(s)

    vertices = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)
    indices = mesh.faces.astype(np.int32).flatten()

    try:
        vc = mesh.visual.vertex_colors
        colors = vc[:, :3].astype(np.float32) / 255.0
    except Exception:
        colors = np.full((len(vertices), 3), 0.7, dtype=np.float32)

    n_v = len(vertices)
    n_t = len(indices) // 3
    print(f"{n_v} verts, {n_t} tris")

    return MeshData(vertices, normals, colors, indices, name)


# ─────────────────────────────────────────────────────
# Renderer class
# ─────────────────────────────────────────────────────

class OpenGLRenderer:
    """Off-screen renderer with instant model swapping."""

    def __init__(
        self,
        mesh_data: MeshData,
        width: int = config.RENDER_WIDTH,
        height: int = config.RENDER_HEIGHT,
    ):
        self.width = width
        self.height = height
        self._shadow_radius = config.MARKER_LENGTH_M * 1.2

        # ── GL context ──
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
        )

        # ── Main program ──
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        # ── Shadow program ──
        self.shadow_prog = self.ctx.program(
            vertex_shader=SHADOW_VERTEX,
            fragment_shader=SHADOW_FRAGMENT,
        )
        sv, si = _make_shadow_quad(self._shadow_radius)
        self.shadow_vao = self.ctx.vertex_array(
            self.shadow_prog,
            [(self.ctx.buffer(sv.tobytes()), "3f", "in_position")],
            index_buffer=self.ctx.buffer(si.tobytes()),
            index_element_size=4,
        )

        # ── FBO ──
        self.color_att = self.ctx.texture((width, height), 4)
        self.depth_att = self.ctx.depth_renderbuffer((width, height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_att],
            depth_attachment=self.depth_att,
        )

        # ── Upload first model ──
        self.vao = None
        self._bufs = []
        self._upload(mesh_data)

        print(f"[opengl_renderer] Ready  ({width}x{height})")

    def _upload(self, md: MeshData):
        """Upload pre-processed mesh data to GPU buffers (fast)."""
        # Release old buffers if any
        if self.vao is not None:
            self.vao.release()
        for b in self._bufs:
            b.release()

        vbo_pos  = self.ctx.buffer(md.vertices.tobytes())
        vbo_norm = self.ctx.buffer(md.normals.tobytes())
        vbo_col  = self.ctx.buffer(md.colors.tobytes())
        ibo      = self.ctx.buffer(md.indices.tobytes())

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (vbo_pos,  "3f", "in_position"),
                (vbo_norm, "3f", "in_normal"),
                (vbo_col,  "3f", "in_color"),
            ],
            index_buffer=ibo,
            index_element_size=4,
        )
        self._bufs = [vbo_pos, vbo_norm, vbo_col, ibo]

    def swap_model(self, mesh_data: MeshData):
        """Instantly swap to a different pre-loaded model (GPU upload only)."""
        self._upload(mesh_data)

    def set_wireframe(self, enabled: bool):
        """Toggle wireframe rendering."""
        self._wireframe = enabled

    def render(self, projection: np.ndarray, model_view: np.ndarray) -> np.ndarray:
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        mvp = (projection @ model_view).astype(np.float32)
        mv  = model_view.astype(np.float32)
        nmat = np.linalg.inv(mv[:3, :3]).T.astype(np.float32)

        # Shadow
        self.ctx.depth_mask = False
        self.shadow_prog["u_mvp"].write(mvp.T.tobytes())
        self.shadow_prog["u_radius"].value = self._shadow_radius
        self.shadow_vao.render(moderngl.TRIANGLES)
        self.ctx.depth_mask = True

        # Model
        self.prog["u_mvp"].write(mvp.T.tobytes())
        self.prog["u_mv"].write(mv.T.tobytes())
        self.prog["u_normal_mat"].write(nmat.T.tobytes())
        if getattr(self, '_wireframe', False):
            self.ctx.wireframe = True
        self.vao.render(moderngl.TRIANGLES)
        if getattr(self, '_wireframe', False):
            self.ctx.wireframe = False

        raw = self.fbo.read(components=4, alignment=1)
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height, self.width, 4,
        )
        return np.flipud(rgba).copy()

    def release(self):
        if self.vao:
            self.vao.release()
        for b in self._bufs:
            b.release()
        self.shadow_vao.release()
        self.fbo.release()
        self.color_att.release()
        self.depth_att.release()
        self.prog.release()
        self.shadow_prog.release()
        self.ctx.release()
        print("[opengl_renderer] Released.")
