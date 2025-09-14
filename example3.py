import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
from scipy.spatial import KDTree
import pandas as pd
from scipy.stats import gaussian_kde
import trimesh


def generate_samples_kde(column, n_samples, bw_method, bw_multiplier=0.2, min_value=0.05):
    """
    Generate random samples from a KDE distribution
    """
    # Ensure input is a 1D numpy array
    data = np.asarray(column).ravel()
    
    # Fit KDE with scipy.stats.gaussian_kde
    kde = gaussian_kde(data, bw_method=bw_method)
    
    # Apply bandwidth multiplier if specified
    if bw_multiplier != 1.0:
        current_bandwidth = kde.factor * bw_multiplier
        kde.set_bandwidth(current_bandwidth)
    
    # Sample from KDE until we have enough valid samples
    valid_samples = []
    samples_needed = n_samples
    
    while samples_needed > 0:
        # Generate samples in batches
        batch_samples = kde.resample(samples_needed * 2).ravel()  # Generate extra to account for filtering
        
        # Filter samples that meet the minimum value requirement
        valid_batch = batch_samples[batch_samples > min_value]
        
        # Add valid samples to our collection
        valid_samples.extend(valid_batch)
        samples_needed = n_samples - len(valid_samples)
    
    # Take exactly n_samples (in case we got more than needed)
    samples = np.array(valid_samples[:n_samples])
    
    return samples, kde

def load_stl_model(stl_filename):
    """
    Load STL model and return vertices, faces, and bounding box
    """
    stl_mesh = mesh.Mesh.from_file(stl_filename)
    
    # Get all vertices and reshape to get unique ones
    all_vertices = stl_mesh.vectors.reshape(-1, 3)
    unique_vertices, unique_indices = np.unique(all_vertices, axis=0, return_inverse=True)
    stl_faces = unique_indices.reshape(-1, 3)
    
    # Get bounding box of the STL model
    min_bounds = np.min(unique_vertices, axis=0)
    max_bounds = np.max(unique_vertices, axis=0)
    
    return unique_vertices, stl_faces, min_bounds, max_bounds

def is_point_inside_mesh(point, vertices, faces):
    """
    Check if a point is inside a mesh using ray casting algorithm
    This is a simplified version - for production use consider more robust methods
    """
    # Simple bounding box check first
    if (point[0] < np.min(vertices[:,0]) or point[0] > np.max(vertices[:,0]) or
        point[1] < np.min(vertices[:,1]) or point[1] > np.max(vertices[:,1]) or
        point[2] < np.min(vertices[:,2]) or point[2] > np.max(vertices[:,2])):
        return False
    
    # For a quick demo, we'll use a simple approach
    # In practice, you might want to use trimesh or other libraries for robust point-in-mesh testing
    return True

def generate_fiber_stl(stl_vertices, stl_faces, lengths, diameters, orientation=None):
    """
    Generate random fibers within an STL model boundary with random lengths and diameters
    """
    N = len(lengths)
    fiber = np.zeros((N, 7))  # Now 7 columns: x1,y1,z1,x2,y2,z2,diameter
    min_bounds = np.min(stl_vertices, axis=0)
    max_bounds = np.max(stl_vertices, axis=0)
    
    for i in range(N):
        attempts = 0
        L = lengths[i]  # Get length for this fiber
        fiber_diameter = diameters[i]  # Get diameter for this fiber
        
        while attempts < 1000:
            # Generate random starting point within the bounding box
            x1 = min_bounds[0] + (max_bounds[0] - min_bounds[0]) * np.random.rand()
            y1 = min_bounds[1] + (max_bounds[1] - min_bounds[1]) * np.random.rand()
            z1 = min_bounds[2] + (max_bounds[2] - min_bounds[2]) * np.random.rand()
            
            start_point = np.array([x1, y1, z1])
            
            # Check if starting point is inside mesh
            if not is_point_inside_mesh(start_point, stl_vertices, stl_faces):
                attempts += 1
                continue
            
            if orientation is None:
                # Random orientation
                theta = 2 * np.pi * np.random.rand()
                v = 2 * np.random.rand() - 1
                
                direction = np.array([
                    np.sqrt(1 - v**2) * np.cos(theta),
                    np.sqrt(1 - v**2) * np.sin(theta),
                    v
                ])
            else:
                # Fixed orientation
                direction = np.array(orientation) / np.linalg.norm(orientation)
                if np.random.rand() > 0.5:
                    direction = -direction
            
            # Calculate end point
            end_point = start_point + L * direction
            
            # Check if end point is inside mesh
            if is_point_inside_mesh(end_point, stl_vertices, stl_faces):
                fiber[i, 0:3] = start_point
                fiber[i, 3:6] = end_point
                fiber[i, 6] = fiber_diameter  # Store diameter
                break
            
            attempts += 1
        
        if attempts >= 1000:
            print(f"Warning: Could not place fiber {i} after 1000 attempts")
    
    return fiber

def plot_mesh_stl(stl_vertices, stl_faces, fibers, scale_factor, nodes, tetrahedrons):
    """
    Plot fibers as cylinders and tetrahedral mesh with STL model as boundary
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- Plot STL model ---
    stl_collection = Poly3DCollection(stl_vertices[stl_faces], alpha=0.1, linewidths=0.5)
    stl_collection.set_facecolor('lightgray')
    stl_collection.set_edgecolor('black')
    ax.add_collection3d(stl_collection)
    
    # --- Plot fibers as cylinders ---
    for i in range(fibers.shape[0]):
        start_point = fibers[i, 0:3]
        end_point = fibers[i, 3:6]
        diameter = fibers[i, 6] * scale_factor  # Apply scale factor to diameter
        
        # Create cylinder points
        vec = end_point - start_point
        length = np.linalg.norm(vec) * scale_factor
        if length == 0:
            continue
            
        # Create cylinder orientation
        z = np.array([0, 0, 1])
        axis = np.cross(z, vec)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            axis = np.array([1, 0, 0])  # Default axis if vectors are parallel
        else:
            axis = axis / axis_norm
        
        angle = np.arccos(np.dot(z, vec) / length)
        
        # Generate cylinder points
        cylinder_resolution = 8  # Number of sides for the cylinder
        radius = diameter / 2  # Already scaled
        
        # Create base cylinder along Z-axis
        theta = np.linspace(0, 2 * np.pi, cylinder_resolution)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z_cyl = np.linspace(0, length, 2)
        
        X, Z = np.meshgrid(x, z_cyl)
        Y, Z = np.meshgrid(y, z_cyl)
        
        # Rotate and translate cylinder
        for j in range(len(z_cyl)):
            for k in range(cylinder_resolution):
                point = np.array([X[j, k], Y[j, k], Z[j, k]])
                
                # Rotate from Z-axis to fiber direction
                if angle != 0:
                    rot_matrix = rotation_matrix(axis, angle)
                    point = np.dot(rot_matrix, point)
                
                # Translate to start point
                point += start_point
                X[j, k], Y[j, k], Z[j, k] = point
        
        # Plot cylinder surface
        ax.plot_surface(X, Y, Z, alpha=0.7, color='blue', linewidth=0)
    
    # --- Set axis limits based on STL model ---
    min_bounds = np.min(stl_vertices, axis=0)
    max_bounds = np.max(stl_vertices, axis=0)
    
    ax.set_xlim(min_bounds[0], max_bounds[0])
    ax.set_ylim(min_bounds[1], max_bounds[1])
    ax.set_zlim(min_bounds[2], max_bounds[2])
    
    # --- Axis labels ---
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # --- View settings ---
    extents = max_bounds - min_bounds
    ax.set_box_aspect(extents)
    
    plt.tight_layout()
    plt.show()

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def abaqus_frc_mesh_stl(stl_filename, excel_filename, n_samples, orientation, filename, 
                       mesh_density, bw_multiplier, bw_method="silverman"):
    """
    Generate FRC mesh for Abaqus using STL model as boundary with random fiber lengths and diameters
    """
    # Load fiber data from Excel
    df = pd.read_excel(excel_filename)
    
    # Generate random lengths and diameters from KDE distributions
    random_lengths, kde_length = generate_samples_kde(
        df['length'], 
        n_samples=n_samples,
        bw_method=bw_method,
        bw_multiplier=bw_multiplier
    )
    
    random_diameters, kde_diam = generate_samples_kde(
        df['average_diameter'], 
        n_samples=n_samples,
        bw_method=bw_method,
        bw_multiplier=bw_multiplier
    )
    
    # Load STL model
    stl_vertices, stl_faces, min_bounds, max_bounds = load_stl_model(stl_filename)
    
    # Generate fibers within STL boundary with random lengths and diameters
    fibers = generate_fiber_stl(stl_vertices, stl_faces, random_lengths, random_diameters, orientation)
    n_fibers = fibers.shape[0]
    
    # Create fiber nodes and elements
    nodes_fibers = []
    elements_fibers = []
    
    for i in range(n_fibers):
        nodes_fibers.append(fibers[i, 0:3])
        nodes_fibers.append(fibers[i, 3:6])
        elements_fibers.append([2*i + 1, 2*i + 2])  # 1-based indexing
    
    nodes_fibers = np.array(nodes_fibers)
    
    # Create background mesh nodes within STL bounding box
    x_range = max_bounds[0] - min_bounds[0]
    y_range = max_bounds[1] - min_bounds[1]
    z_range = max_bounds[2] - min_bounds[2]
    
    dx = x_range / mesh_density
    dy = y_range / mesh_density
    dz = z_range / mesh_density
    
    x_nodes = np.arange(min_bounds[0], max_bounds[0] + dx, dx)
    y_nodes = np.arange(min_bounds[1], max_bounds[1] + dy, dy)
    z_nodes = np.arange(min_bounds[2], max_bounds[2] + dz, dz)
    
    xx, yy, zz = np.meshgrid(x_nodes, y_nodes, z_nodes)
    node_mesh = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Filter nodes to only include those inside the STL mesh
    node_mesh_inside = []
    for point in node_mesh:
        if is_point_inside_mesh(point, stl_vertices, stl_faces):
            node_mesh_inside.append(point)
    
    node_mesh_inside = np.array(node_mesh_inside)
    
    # Combine all nodes
    nodes = np.vstack([nodes_fibers, node_mesh_inside])
    
    # Generate tetrahedral mesh
    tetrahedrons = Delaunay(nodes).simplices
    n_tetra = tetrahedrons.shape[0]
    
    # Plot the mesh
    plot_mesh_stl(stl_vertices, stl_faces, fibers, scale_factor, nodes, tetrahedrons)
    
    # Prepare element sets for Abaqus
    elements = []
    elements.extend(elements_fibers)
    elements.extend(tetrahedrons.tolist())
    
    element_sets = [
        {
            'Name': 'Fiber',
            'Elements_Type': 'T3D2',
            'Elements': list(range(1, n_fibers + 1))
        },
        {
            'Name': 'Matrix', 
            'Elements_Type': 'C3D4',
            'Elements': list(range(n_fibers + 1, n_fibers + n_tetra + 1))
        }
    ]
    
    # Write to Abaqus input file
    write_abaqus_input(filename, nodes, elements, element_sets, fibers)
    
    return nodes, elements, element_sets, fibers, random_lengths, random_diameters

def write_abaqus_input(filename, nodes, elements, element_sets, fibers):
    """
    Write Abaqus input file with proper 1-based indexing and error checking
    """
    with open(filename, 'w') as f:
        # Header
        f.write("*HEADING\n")
        f.write("Concrete beam with steel reinforcement fibers\n")
        f.write("** Generated by Python script\n")
        f.write("**\n")
        
        # Nodes
        f.write("*NODE, NSET=ALL_NODES\n")
        for i, node in enumerate(nodes, 1):
            f.write(f"{i}, {node[0]:.6f}, {node[1]:.6f}, {node[2]:.6f}\n")
        
        # Fiber elements (T3D2)
        fiber_set = next((es for es in element_sets if es['Name'] == 'Fiber'), None)
        if fiber_set:
            f.write(f"*ELEMENT, TYPE=T3D2, ELSET={fiber_set['Name'].upper()}\n")
            for i, elem_idx in enumerate(fiber_set['Elements'], 1):
                elem = elements[elem_idx - 1]  # Convert to 0-based indexing
                # Check for valid node indices (should be >= 1)
                if min(elem) < 1 or max(elem) > len(nodes):
                    print(f"Warning: Invalid element connectivity for fiber element {i}: {elem}")
                    continue
                f.write(f"{i}, {elem[0]}, {elem[1]}\n")
        
        # Concrete elements (C3D4)
        matrix_set = next((es for es in element_sets if es['Name'] == 'Matrix'), None)
        if matrix_set:
            f.write(f"*ELEMENT, TYPE=C3D4, ELSET={matrix_set['Name'].upper()}\n")
            start_idx = len(fiber_set['Elements']) + 1 if fiber_set else 1
            for i, elem_idx in enumerate(matrix_set['Elements'], start_idx):
                elem = elements[elem_idx - 1]  # Convert to 0-based indexing
                # Check for valid node indices and zero values
                if min(elem) < 1 or max(elem) > len(nodes) or 0 in elem:
                    print(f"Warning: Invalid element connectivity for concrete element {i}: {elem}")
                    continue
                f.write(f"{i}, {elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}\n")
        
        # Material properties - Steel
        f.write("*MATERIAL, NAME=STEEL\n")
        f.write("*ELASTIC\n")
        f.write("210000.0, 0.3\n")
        f.write("*PLASTIC\n")
        f.write("500.0, 0.0\n")
        
        # Material properties - Concrete
        f.write("*MATERIAL, NAME=CONCRETE\n")
        f.write("*ELASTIC\n")
        f.write("30000.0, 0.2\n")
        f.write("*CONCRETE\n")
        f.write("40.0, 0.0\n")
        
        # Section definitions
        if matrix_set:
            f.write("*SOLID SECTION, ELSET=MATRIX, MATERIAL=CONCRETE\n")
            f.write(",\n")
        
        if fiber_set:
            # Fiber sections with individual diameters
            f.write("*BEAM SECTION, ELSET=FIBER, MATERIAL=STEEL, SECTION=CIRC\n")
            for i, fiber in enumerate(fibers, 1):
                diameter = fiber[6]
                f.write(f"{diameter:.6f}\n")
                f.write("0.0, 0.0, 1.0\n")  # Orientation vector
        
        # Simple boundary conditions
        f.write("*NSET, NSET=FIXED_BOTTOM\n")
        # Find bottom nodes (z-min)
        min_z = min(node[2] for node in nodes)
        bottom_nodes = [i+1 for i, node in enumerate(nodes) if abs(node[2] - min_z) < 1e-6]
        # Write in chunks of 16 nodes per line (Abaqus limit)
        for i in range(0, len(bottom_nodes), 16):
            chunk = bottom_nodes[i:i+16]
            f.write(", ".join(map(str, chunk)) + "\n")
        
        f.write("*BOUNDARY\n")
        f.write("FIXED_BOTTOM, ENCASTRE\n")
        
        # Step definition
        f.write("*STEP, NAME=LOADING\n")
        f.write("*STATIC\n")
        f.write("1.0, 1.0, 1e-5, 1.0\n")
        
        # Output requests
        f.write("*OUTPUT, FIELD\n")
        f.write("*NODE OUTPUT\n")
        f.write("U, RF\n")
        f.write("*ELEMENT OUTPUT\n")
        f.write("S, E, PE\n")
        
        f.write("*END STEP\n")
    
    print(f"Abaqus input file written to: {filename}")

def generate_concrete_beam_mesh(stl_filename, output_filename, mesh_size):
    """
    Generate a concrete beam mesh from an STL file for Abaqus with proper geometry handling
    
    Parameters:
    -----------
    stl_filename : str
        Path to the STL file
    output_filename : str
        Path for the output Abaqus input file
    mesh_size : float
        Desired mesh size (smaller values = finer mesh)
    """
    # Load STL model using trimesh for better geometry handling
    tri_mesh = trimesh.load(stl_filename)
    
    # Use trimesh to generate a volume mesh that respects the geometry
    # This will properly handle slits and other internal features
    volume_mesh = tri_mesh.voxelized(pitch=mesh_size).marching_cubes
    
    # Get vertices and faces from the volume mesh
    vertices = volume_mesh.vertices
    faces = volume_mesh.faces
    
    # Convert to tetrahedral mesh using Delaunay triangulation
    tetrahedrons = Delaunay(vertices).simplices
    
    # Write Abaqus input file
    with open(output_filename, 'w') as f:
        # Header
        f.write("*HEADING\n")
        f.write("Concrete beam mesh generated from STL\n")
        f.write("** Generated by Python script\n")
        f.write("**\n")
        
        # Nodes
        f.write("*NODE, NSET=ALL_NODES\n")
        for i, node in enumerate(vertices, 1):
            f.write(f"{i}, {node[0]:.6f}, {node[1]:.6f}, {node[2]:.6f}\n")
        
        # Elements
        f.write("*ELEMENT, TYPE=C3D4, ELSET=CONCRETE\n")
        for i, elem in enumerate(tetrahedrons, 1):
            # Convert to 1-based indexing
            elem_1based = elem + 1
            f.write(f"{i}, {elem_1based[0]}, {elem_1based[1]}, {elem_1based[2]}, {elem_1based[3]}\n")
        
        # Material properties - Concrete
        f.write("*MATERIAL, NAME=CONCRETE\n")
        f.write("*ELASTIC\n")
        f.write("30000.0, 0.2\n")
        f.write("*CONCRETE\n")
        f.write("40.0, 0.0\n")
        
        # Section definition
        f.write("*SOLID SECTION, ELSET=CONCRETE, MATERIAL=CONCRETE\n")
        f.write(",\n")
        
        # Simple boundary conditions (fix bottom nodes)
        f.write("*NSET, NSET=FIXED_BOTTOM\n")
        min_z = min(node[2] for node in vertices)
        bottom_nodes = [i+1 for i, node in enumerate(vertices) if abs(node[2] - min_z) < 1e-6]
        
        # Write in chunks of 16 nodes per line (Abaqus limit)
        for i in range(0, len(bottom_nodes), 16):
            chunk = bottom_nodes[i:i+16]
            f.write(", ".join(map(str, chunk)) + "\n")
        
        f.write("*BOUNDARY\n")
        f.write("FIXED_BOTTOM, ENCASTRE\n")
        
        # Step definition
        f.write("*STEP, NAME=LOADING\n")
        f.write("*STATIC\n")
        f.write("1.0, 1.0, 1e-5, 1.0\n")
        
        # Output requests
        f.write("*OUTPUT, FIELD\n")
        f.write("*NODE OUTPUT\n")
        f.write("U, RF\n")
        f.write("*ELEMENT OUTPUT\n")
        f.write("S, E, PE\n")
        
        f.write("*END STEP\n")
    
    print(f"Abaqus input file written to: {output_filename}")
    print(f"Generated {len(vertices)} nodes and {len(tetrahedrons)} elements")
    
    return vertices, tetrahedrons

if __name__ == "__main__":
    # Parameters for STL-based mesh generation
    stl_filename = "3dmodel/3dmodel.stl"
    excel_filename = "steel_fibers/steel_fibers.xlsx"
    n_samples = 10000  # Number of fibers
    orientation = None  # Random orientation
    filename = r"C:/Users/marci/OneDrive/Documents/Abaqus/abaqus_beam.inp"
    mesh_density = 10.0  # Controls the background mesh resolution
    bw_multiplier = 0.2  # KDE bandwidth multiplier
    bw_method = "silverman"  # KDE bandwidth method
    scale_factor = 1  # For visualisation
    
    nodes, elements = generate_concrete_beam_mesh(stl_filename, filename, mesh_density)
    
