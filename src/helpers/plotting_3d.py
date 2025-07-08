import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# making detector boundary points

tpc_min_x = -1.
tpc_max_x = 254.3
tpc_min_y = -115.
tpc_max_y = 117.
tpc_min_z = 0.6
tpc_max_z = 1036.4


def generate_box_wireframe(x_min, x_max, y_min, y_max, z_min, z_max):
    """Generate wireframe lines for a 3D box"""
    
    # Define the 8 vertices of the box
    vertices = np.array([
        [x_min, y_min, z_min],  # 0
        [x_max, y_min, z_min],  # 1
        [x_max, y_max, z_min],  # 2
        [x_min, y_max, z_min],  # 3
        [x_min, y_min, z_max],  # 4
        [x_max, y_min, z_max],  # 5
        [x_max, y_max, z_max],  # 6
        [x_min, y_max, z_max],  # 7
    ])
    
    # Define the 12 edges of the box
    edges = [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    return vertices, edges

def plot_event(event_index, points_dic):


    # Generate wireframe for detector boundaries
    detector_vertices, detector_edges = generate_box_wireframe(tpc_min_x, tpc_max_x, tpc_min_y, tpc_max_y, tpc_min_z, tpc_max_z)
    
    x_width = tpc_max_x - tpc_min_x
    expanded_detector_vertices, expanded_detector_edges = generate_box_wireframe(
        tpc_min_x - x_width, tpc_max_x + x_width, tpc_min_y, tpc_max_y, tpc_min_z, tpc_max_z
    )

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

    # Add expanded detector boundary wireframe (for camera positioning)
    for edge in expanded_detector_edges:
        start_vertex = expanded_detector_vertices[edge[0]]
        end_vertex = expanded_detector_vertices[edge[1]]
        
        fig.add_trace(go.Scatter3d(
            x=[start_vertex[2], end_vertex[2]],
            y=[start_vertex[0], end_vertex[0]],
            z=[start_vertex[1], end_vertex[1]],
            mode='lines',
            line=dict(
                color='black',
                width=1
            ),
            opacity=0,
            showlegend=False,
            visible=True
        ))

    # Add detector boundary wireframe
    for edge in detector_edges:
        start_vertex = detector_vertices[edge[0]]
        end_vertex = detector_vertices[edge[1]]
        
        fig.add_trace(go.Scatter3d(
            x=[start_vertex[2], end_vertex[2]],
            y=[start_vertex[0], end_vertex[0]],
            z=[start_vertex[1], end_vertex[1]],
            mode='lines',
            line=dict(
                color='black',
                width=2,
            ),
            name='TPC Boundary',
            showlegend=False
        ))

    for name, values in points_dic.items():
        
        points, color, cmap, size, visible_by_default = values

        if points.shape == (3,):
            fig.add_trace(go.Scatter3d(
                x=[points[2]],
                y=[points[0]],
                z=[points[1]],
                mode='markers',
                marker=dict(size=size, color=color, opacity=1, colorscale=cmap),
                name=name,
                visible=visible_by_default
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=points[:, 2],
                y=points[:, 0],
                z=points[:, 1],
                mode='markers',
                marker=dict(size=size, color=color, opacity=1, colorscale=cmap),
                name=name,
                visible=visible_by_default
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='z',
            yaxis_title='x',
            zaxis_title='y',
            aspectratio=dict(
                x=5,
                y=3,
                z=1
            ),
        ),
        width=2000,
        height=1200,
        autosize=False,
        scene_camera=dict(
            eye=dict(x=-1.5, y=-1.5, z=1.5)
        )
    )

    fig.show(renderer="browser")

