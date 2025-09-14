
import argparse
import gmsh
import math
import os

def build_geometry_3d(L, W, H, sx, sy, sz, cx, cy, cz):

    occ = gmsh.model.occ
    gmsh.model.add("beam_with_slit_3d")

    # Main solid beam
    beam = occ.addBox(0, -W/2.0, -H/2.0, L, W, H)

    # Void block (slit). If any slit dimension is <= 0, skip subtraction.
    voids = []
    if sx > 0 and sy > 0 and sz > 0:
        vx = cx - sx/2.0
        vy = cy - sy/2.0
        vz = cz - sz/2.0
        v = occ.addBox(vx, vy, vz, sx, sy, sz)
        voids.append((3, v))

    occ.synchronize()

    tool = voids if voids else []
    if tool:
        out, _ = occ.cut(objectDimTags=[(3, beam)], toolDimTags=tool, removeObject=True, removeTool=True)
        occ.synchronize()
        # Return all resulting volumes (could be more than one if slit splits the body)
        vols = [tag for (dim, tag) in out if dim == 3]
    else:
        vols = [beam]

    return vols

def build_geometry_2d(L, W, sx, sy, cx, cy):
    """
    Build a 2D rectangular plate (in XY) with a rectangular hole (slit).
    Plate spans x in [0, L], y in [-W/2, W/2]
    """
    occ = gmsh.model.occ
    gmsh.model.add("beam_with_slit_2d")

    plate = occ.addRectangle(0, -W/2.0, 0.0, L, W)
    voids = []
    if sx > 0 and sy > 0:
        vx = cx - sx/2.0
        vy = cy - sy/2.0
        v = occ.addRectangle(vx, vy, 0.0, sx, sy)
        voids.append((2, v))

    occ.synchronize()

    if voids:
        out, _ = occ.cut(objectDimTags=[(2, plate)], toolDimTags=voids, removeObject=True, removeTool=True)
        occ.synchronize()
        faces = [tag for (dim, tag) in out if dim == 2]
    else:
        faces = [plate]

    return faces

def write_inp_from_gmsh(out_inp_path, mesh_dim=3, etype2d="S3", etype3d="C3D4",
                        material_name="Concrete", E=3e10, nu=0.2):
    # Nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = node_tags.astype(int)
    nodes_xyz = node_coords.reshape(-1, 3)

    # Remap node tags to contiguous 1..N
    tag_to_new = {int(t): i+1 for i, t in enumerate(node_tags.tolist())}
    idx_by_old = {int(t): i for i, t in enumerate(node_tags.tolist())}

    # Elements
    types, all_elem_tags, all_elem_node_tags = gmsh.model.mesh.getElements(mesh_dim)
    elem_blocks = {"S3": [], "CPS3": [], "C3D4": [], "C3D10": []}

    for etype, tags, conn in zip(types, all_elem_tags, all_elem_node_tags):
        name, dim, order, nper, param_dim, *_ = gmsh.model.mesh.getElementProperties(etype)

        # Map to Abaqus types we support
        if mesh_dim == 2:
            if nper == 3:
                abaqus_type = etype2d  # S3 or CPS3
            else:
                continue  # skip quads / higher order for simplicity
        else:  # 3D
            if nper == 4:
                abaqus_type = "C3D4"
            elif nper == 10:
                abaqus_type = "C3D10"
            else:
                continue  # skip other types

        conns = [conn[i:i+nper] for i in range(0, len(conn), nper)]
        for c in conns:
            elem_blocks[abaqus_type].append([tag_to_new[int(nt)] for nt in c])

    with open(out_inp_path, "w") as f:
        f.write("*Heading\n")
        f.write("** Procedural beam with slit, meshed via Gmsh\n")

        # Nodes
        f.write("*Node\n")
        for old_tag, new_id in sorted(tag_to_new.items(), key=lambda kv: kv[1]):
            ix = idx_by_old[old_tag]
            x, y, z = nodes_xyz[ix]
            f.write(f"{new_id}, {x:.9g}, {y:.9g}, {z:.9g}\n")

        # Elements (grouped by type)
        eid = 1
        total = 0
        for typ in ("S3", "CPS3", "C3D4", "C3D10"):
            block = elem_blocks[typ]
            if not block:
                continue
            f.write(f"*Element, type={typ}\n")
            for conn in block:
                f.write(f"{eid}, " + ", ".join(map(str, conn)) + "\n")
                eid += 1
                total += 1

        if total > 0:
            f.write("*Elset, elset=ALL, generate\n")
            f.write(f"1, {total}, 1\n")

        # Material & Section
        f.write(f"*Material, name={material_name}\n")
        f.write("*Elastic\n")
        f.write(f"{E}, {nu}\n")
        if mesh_dim == 3:
            f.write(f"*Solid Section, material={material_name}, elset=ALL\n")
        else:
            f.write(f"*Shell Section, material={material_name}, elset=ALL, thickness=1.0\n")

        # Minimal step
        f.write("*Step, name=LoadStep, nlgeom=NO\n")
        f.write("*Static\n1., 1., 1e-05, 1.\n")
        f.write("** Add *Boundary / *Cload here\n")
        f.write("*End Step\n")

def run(L, W, H, sx, sy, sz, cx, cy, cz, dim, mesh_size, etype2d, etype3d, out_path):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)

        if dim == 3:
            _ = build_geometry_3d(L, W, H, sx, sy, sz, cx, cy, cz)
            gmsh.model.occ.synchronize()
        else:
            _ = build_geometry_2d(L, W, sx, sy, cx, cy)
            gmsh.model.occ.synchronize()

        # Mesh controls
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(mesh_size))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(mesh_size))

        # Element order for 3D if C3D10 requested
        if dim == 3 and etype3d.upper() == "C3D10":
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            # Keep full quadratic (not serendipity)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        else:
            gmsh.option.setNumber("Mesh.ElementOrder", 1)

        gmsh.model.mesh.generate(dim)

        write_inp_from_gmsh(out_path, mesh_dim=dim, etype2d=etype2d.upper(), etype3d=etype3d.upper())
        print(f"✅ Wrote Abaqus input: {out_path} (dim={dim}, size={mesh_size}, etype2d={etype2d}, etype3d={etype3d})")
    finally:
        gmsh.finalize()

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser(description="Rectangular beam with rectangular slit → Abaqus INP via Gmsh")
    # Beam (outer) dimensions
    ap.add_argument("--L",  type=float, default=600.0, help="Beam length (X)")
    ap.add_argument("--W",  type=float, default=150.0, help="Beam width (Y)")
    ap.add_argument("--H",  type=float, default=150.0, help="Beam height (Z)")

    # Slit: runs along the shorter side (Y), at base; cross-section 5 (X) × 20 (Z)
    ap.add_argument("--sx", type=float, default=5.0,   help="Slit size in X (width of slit)")
    ap.add_argument("--sy", type=float, default=20.0, help="Slit size in Y (runs full width)")
    ap.add_argument("--sz", type=float, default=150.0,  help="Slit size in Z (height)")

    # Slit center: mid-length (X=L/2), centered in Y, sitting on base in Z
    ap.add_argument("--cx", type=float, default=300.0, help="Slit center X (L/2)")
    ap.add_argument("--cy", type=float, default=-65.0,   help="Slit center Y (0)")
    ap.add_argument("--cz", type=float, default=-0.0, help="Slit center Z = -H/2 + sz/2")
    # Mesh + element options
    ap.add_argument("--dim",     type=int, choices=[2, 3], default=3, help="2=surface/plane mesh, 3=solid volume mesh")
    ap.add_argument("--mesh",    type=float, default=10.0, help="Target element size")
    ap.add_argument("--etype2d", default="S3", choices=["S3", "CPS3"], help="Abaqus 2D element type")
    ap.add_argument("--etype3d", default="C3D4", choices=["C3D4", "C3D10"], help="Abaqus 3D element type")
    ap.add_argument("--outdir",  default="C:/Users/marci/OneDrive/Documents/Abaqus/6.inp", help="Directory where .inp will be saved")

    args = ap.parse_args()

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True,)

    # Build a filename based on parameters
    base = f"beam_L{int(args.L)}_W{int(args.W)}_H{int(args.H)}_mesh{int(args.mesh)}.inp"
    out_path = os.path.join(args.outdir, base)

    cx = args.cx if args.cx is not None else args.L / 2.0  # default to mid-length

    run(L=args.L, W=args.W, H=args.H,
        sx=args.sx, sy=args.sy, sz=args.sz,
        cx=cx, cy=args.cy, cz=args.cz,
        dim=args.dim, mesh_size=args.mesh,
        etype2d=args.etype2d, etype3d=args.etype3d,
        out_path=out_path)
    
    
