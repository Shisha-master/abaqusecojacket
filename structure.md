1) Configuration & Inputs

Units & model name

Beam geometry: length, width, height

Fiber specs:

List(s) of lengths and diameters (or probability distributions)

Volume fraction target (or total fiber count)

Orientation distribution (fully random 3D, planar, or biased)

Minimum edge clearance and inter-fiber spacing (no overlap)

Materials:

Concrete: elastic + (optional) Concrete Damaged Plasticity (CDP) params

Steel fiber: elastic–plastic params

(Optional) Interface: cohesive/traction–separation or perfect bond

Meshing:

Global element size + refinements around fibers

Element types (C3D8R, C3D4, etc.) and fiber element type (B31/T3D2/solid)

Boundary conditions & loads

e.g., three-point or four-point bending, end displacement, etc.

Random seed for reproducibility

Output controls (history fields, field outputs, images)

2) Model Bootstrapping

Create Model, Sketch, and Part for the beam (3D deformable solid).

Create Materials and Sections:

Concrete material + section (solid)

Steel material + section (beam/truss/solid)

(Optional) Cohesive interaction properties

Create Assembly and a Datum CSYS for convenience.

3) Fiber Generation (Geometry & Placement)

Compute target number of fibers from volume fraction (or use given count).

Sampling loop to place fibers:

Sample center position uniformly inside a feasible domain (beam bbox shrunk by edge clearance).

Sample orientation (unit vector) according to desired distribution.

Sample length and diameter per your lists/distributions.

Build the endpoints: center ± 0.5 * length * direction.

Reject if:

Any endpoint violates clearance to faces.

Segment intersects beam boundary given diameter.

Violates minimum distance to existing fibers (no overlap).

Acceleration structures for overlap checks:

Keep a list of fiber center points and use distance thresholds.

(Optional) Use a simple grid/binning map to reduce pairwise checks.

Save final accepted fibers as records: {id, x1, x2, d, L, dir}.

4) Fiber Representation Strategy (choose one)

Discrete beam/truss elements (recommended):

Create a wire Part (or multiple instances) for each fiber line segment.

Assign beam/truss section with the sampled diameter.

Use Embedded Element technique:

Host: concrete solid; Embedded: fiber wires.

Or use Tie constraints along lines if meshed compatibly.

Discrete 3D solid fibers (costly):

Sweep circular profiles along segments → boolean merge or tie/embedded.

Homogenized material (alternative):

Skip explicit fibers; use an RVE-based effective concrete+fiber material (not your case if you need discrete fibers).

5) Partitioning & Meshing

Partition beam if needed (seed control zones, supports/loading regions).

Seed mesh:

Global size for concrete; refined size around embedded wires if possible.

Mesh element types:

Concrete: C3D8R (or C3D4 for rough, C3D10 for higher fidelity).

Fibers: B31 (beam) or T3D2 (truss).

Generate meshes for both host and embedded parts.

6) Interactions & Constraints

Embedded region: Assign all fiber element sets as embedded in the concrete.

(Optional) Cohesive laws:

If modeling debonding, replace embedded with surface-to-surface interactions along fiber surfaces (requires solids) or custom connector elements—more complex & heavier.

7) Steps, BCs, and Loads

Static General (quasi-static) or Dynamic/Implicit if needed for stability.

Boundary conditions:

E.g., simply supported: one end pin (U1=U2=U3=0), other end roller (U2=U3=0).

If 3-/4-point bending: add rigid bodies (reference points + coupling) for supports and loading noses.

Loads:

Concentrated force or displacement at loading nose RP(s).

Load ramp with amplitude to avoid snap-through issues.

8) Output Requests

Field output: S, E, U, PE, PEEQ, DAMAGET/DAMAGEC (if CDP), contact variables, section forces in beams.

History output:

Reaction force at supports, load RP displacement, force–deflection curve.

Optional: energy terms for stability diagnostics.

9) Post-Processing (in-script)

Extract:

Load–deflection data at the midspan RP.

Crack/damage indicators (if CDP) distributions.

Fiber axial force histograms.

Save CSV/JSON + generate simple plots (optional).

10) Validation & Safety Checks

Check achieved volume fraction vs target.

Check fiber length & diameter distribution matches input stats.

Check min distance and edge clearance constraints satisfied.

Check mesh size vs fiber diameter recommendations.

Log random seed and accept/reject counts.

11) Script Organization (files & functions)

main.py — orchestrates the workflow.

config.yaml/json — all user inputs & distributions.

geom.py

make_beam_part(model, L, W, H)

make_supports_and_noses(...) (if using RPs/rigids)

materials.py

define_concrete(model, params)

define_steel(model, params)

assign_sections(...)

fibers.py

sample_orientation(dist_params)

sample_length_diameter(...)

propose_fiber(center, dir, L)

overlap_check(fiber, spatial_index, min_gap)

create_fiber_wire_part(...)

embed_fibers_in_host(...)

mesh.py

seed_and_mesh_concrete(...)

seed_and_mesh_fibers(...)

bc_loads.py

create_steps(...)

apply_bcs(...)

apply_loads(...)

output.py

request_outputs(...)

postprocess_to_csv_odb(...)

utils.py

rng(seed)

bbox_and_clearances(...)

spatial_hashing(...)

logger(...)

12) Inputs you’ll want to define precisely

Orientation model:

Uniform on sphere (random 3D): sample azimuth ∈ [0,2π), cosθ ∈ [−1,1].

Planar random: fix one component, random angle in plane.

Biased alignment: von Mises–Fisher or weighted interpolation to a preferred axis.

Distributions:

Deterministic sets you “know” (choose by index) or parametric (normal/lognormal) with truncation.

Spacing constraints:

Minimum centerline–centerline distance ≥ 0.5*(d_i + d_j) + gap.

Endpoint margin ≥ 0.5*d + edge_clearance.

Counts:

From volume fraction Vf: N ≈ Vf * V_beam / V_fiber_avg with V_fiber = π (d/2)^2 * L.

13) Performance tips

Generate fibers once as geometry, then instance in the assembly (lighter).

Use spatial hashing (voxel grid) for O(1) neighborhood checks.

Start with a looser min gap to populate, then tighten (optional two-pass).

Consider truss fibers first (fast) before moving to beam sections with orientations & profiles.

14) Optional extensions

Curved/segmented fibers (polyline with slight waviness).

Fiber length distributions by tier (e.g., two populations).

Damage/plasticity calibration via small RVE submodels.

Monte Carlo batches (loop multiple seeds, aggregate stats).