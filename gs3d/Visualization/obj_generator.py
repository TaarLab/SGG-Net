import numpy as np

def generate_hourglass_obj(file_name="hourglass.obj", top_radius=1.0, bottom_radius=1.0, waist_radius=0.2, height=2.0, landing_radius=1.5, landing_height=0.1, num_segments=32, pile_radius=0.1, pile_height=2.2, num_piles=3):
    # Top and bottom positions
    top_y = height / 2.0
    bottom_y = -height / 2.0
    waist_y = 0.0

    vertices = []
    faces = []

    # Generate vertices for the top circle
    for i in range(num_segments):
        angle = 2.0 * np.pi * i / num_segments
        x = top_radius * np.cos(angle)
        z = top_radius * np.sin(angle)
        vertices.append((x, top_y, z))

    # Generate vertices for the waist circle
    for i in range(num_segments):
        angle = 2.0 * np.pi * i / num_segments
        x = waist_radius * np.cos(angle)
        z = waist_radius * np.sin(angle)
        vertices.append((x, waist_y, z))

    # Generate vertices for the bottom circle
    for i in range(num_segments):
        angle = 2.0 * np.pi * i / num_segments
        x = bottom_radius * np.cos(angle)
        z = bottom_radius * np.sin(angle)
        vertices.append((x, bottom_y, z))

    # Add the top center vertex
    top_center = (0.0, top_y, 0.0)
    vertices.append(top_center)

    # Add the bottom center vertex
    bottom_center = (0.0, bottom_y, 0.0)
    vertices.append(bottom_center)

    # Generate faces between top circle and waist circle
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append((i + 1, next_i + 1, num_segments + next_i + 1))
        faces.append((i + 1, num_segments + next_i + 1, num_segments + i + 1))

    # Generate faces between waist circle and bottom circle
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append((num_segments + i + 1, num_segments + next_i + 1, 2 * num_segments + next_i + 1))
        faces.append((num_segments + i + 1, 2 * num_segments + next_i + 1, 2 * num_segments + i + 1))

    # Generate faces for the top cap
    top_center_index = len(vertices) - 2
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append((top_center_index + 1, next_i + 1, i + 1))

    # Generate faces for the bottom cap
    bottom_center_index = len(vertices) - 1
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append((2 * num_segments + i + 1, 2 * num_segments + next_i + 1, bottom_center_index + 1))

    ### Create 3D landing parts with height

    # Top landing (with height) - vertices
    for i in range(num_segments):
        angle = 2.0 * np.pi * i / num_segments
        x = landing_radius * np.cos(angle)
        z = landing_radius * np.sin(angle)
        # Top surface of the landing
        vertices.append((x, top_y + landing_height, z))
        # Bottom surface of the landing
        vertices.append((x, top_y, z))

    # Add top landing center vertices
    top_landing_center_index_upper = len(vertices)
    vertices.append((0.0, top_y + landing_height, 0.0))  # Top surface center
    top_landing_center_index_lower = len(vertices)
    vertices.append((0.0, top_y, 0.0))  # Bottom surface center

    # Generate faces for the top landing (upper and lower surfaces + side walls)
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        # Top surface
        faces.append((top_landing_center_index_upper + 1, len(vertices) - 2 * num_segments + i * 2, len(vertices) - 2 * num_segments + next_i * 2))
        # Bottom surface
        faces.append((top_landing_center_index_lower + 1, len(vertices) - 2 * num_segments + next_i * 2 + 1, len(vertices) - 2 * num_segments + i * 2 + 1))

        # Side walls connecting top and bottom surfaces of the landing
        faces.append((len(vertices) - 2 * num_segments + i * 2, len(vertices) - 2 * num_segments + next_i * 2, len(vertices) - 2 * num_segments + next_i * 2 + 1))
        faces.append((len(vertices) - 2 * num_segments + i * 2, len(vertices) - 2 * num_segments + next_i * 2 + 1, len(vertices) - 2 * num_segments + i * 2 + 1))

    # Bottom landing (with height) - vertices
    for i in range(num_segments):
        angle = 2.0 * np.pi * i / num_segments
        x = landing_radius * np.cos(angle)
        z = landing_radius * np.sin(angle)
        # Bottom surface of the landing
        vertices.append((x, bottom_y - landing_height, z))
        # Top surface of the landing
        vertices.append((x, bottom_y, z))

    # Add bottom landing center vertices
    bottom_landing_center_index_lower = len(vertices)
    vertices.append((0.0, bottom_y - landing_height, 0.0))  # Bottom surface center
    bottom_landing_center_index_upper = len(vertices)
    vertices.append((0.0, bottom_y, 0.0))  # Top surface center

    # Generate faces for the bottom landing (upper and lower surfaces + side walls)
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        # Bottom surface
        faces.append((bottom_landing_center_index_lower + 1, len(vertices) - 2 * num_segments + next_i * 2, len(vertices) - 2 * num_segments + i * 2))
        # Top surface
        faces.append((bottom_landing_center_index_upper + 1, len(vertices) - 2 * num_segments + i * 2 + 1, len(vertices) - 2 * num_segments + next_i * 2 + 1))

        # Side walls connecting bottom and top surfaces of the landing
        faces.append((len(vertices) - 2 * num_segments + next_i * 2 + 1, len(vertices) - 2 * num_segments + i * 2 + 1, len(vertices) - 2 * num_segments + i * 2))
        faces.append((len(vertices) - 2 * num_segments + next_i * 2 + 1, len(vertices) - 2 * num_segments + i * 2, len(vertices) - 2 * num_segments + next_i * 2))

    ### Cylindrical piles, now positioned inside the landing parts

    pile_top_y = top_y + landing_height
    pile_bottom_y = bottom_y - landing_height
    pile_segments = 16  # Number of segments to approximate the cylinder

    for pile_index in range(num_piles):
        pile_angle = 2.0 * np.pi * pile_index / num_piles
        pile_center_x = (landing_radius - pile_radius) * np.cos(pile_angle)
        pile_center_z = (landing_radius - pile_radius) * np.sin(pile_angle)

        # Generate the vertices for the pile at the top and bottom
        for i in range(pile_segments):
            angle = 2.0 * np.pi * i / pile_segments
            x_offset = pile_radius * np.cos(angle)
            z_offset = pile_radius * np.sin(angle)
            vertices.append((pile_center_x + x_offset, pile_top_y, pile_center_z + z_offset))
            vertices.append((pile_center_x + x_offset, pile_bottom_y, pile_center_z + z_offset))

        # Create faces to connect the top and bottom segments of the cylinder
        start_index = len(vertices) - 2 * pile_segments
        for i in range(pile_segments):
            next_i = (i + 1) % pile_segments
            faces.append((start_index + i * 2 + 1, start_index + next_i * 2 + 1, start_index + next_i * 2 + 2))
            faces.append((start_index + i * 2 + 1, start_index + next_i * 2 + 2, start_index + i * 2 + 2))

    # Write to an OBJ file
    with open(file_name, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"OBJ file '{file_name}' generated successfully!")

# Call the function to generate the hourglass
generate_hourglass_obj("hourglass_with_complete_landings.obj")
