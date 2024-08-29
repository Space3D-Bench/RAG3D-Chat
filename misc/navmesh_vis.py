from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import plotly.graph_objects as go


def visualize_navmesh_3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    out_filename: Path,
    waypoints: Optional[np.array] = None,
    points: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Visualizes the navigation mesh in 3D.

    Args:
        vertices (np.ndarray): vertices of the navigation mesh
        faces (np.ndarray): faces of the navigation mesh
        out_filename (Path): path to the output HTML file
        waypoints (optional, np.array): waypoints of the path to be visualized
        points (optional, Dict[str, Dict[str, Any]]): points to be visualized 
            (with their names, colors, and positions)
    
    """
    max_xyz = -float("inf")
    min_xyz = float("inf")
    triangles = []
    edge_x, edge_y, edge_z = [], [], []

    for face in faces:
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]

        triangles.append([v1, v2, v3])
        max_xyz = max(max_xyz, *v1, *v2, *v3)
        min_xyz = min(min_xyz, *v1, *v2, *v3)
        edge_x.extend([v1[0], v2[0], v2[0], v3[0], v3[0], v1[0], None])
        edge_y.extend([v1[1], v2[1], v2[1], v3[1], v3[1], v1[1], None])
        edge_z.extend([v1[2], v2[2], v2[2], v3[2], v3[2], v1[2], None])

    x, y, z = zip(*np.concatenate(triangles))
    i, j, k = zip(*[(i, i + 1, i + 2) for i in range(0, len(x), 3)])
    fig = go.Figure(
        data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="blue", opacity=0.5)]
    )

    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="black", width=2),
            name="edges",
        )
    )

    if waypoints is not None:
        waypoint_x, waypoint_y, waypoint_z = zip(*waypoints)
        fig.add_trace(
            go.Scatter3d(
                x=waypoint_x,
                y=waypoint_y,
                z=waypoint_z,
                mode="lines",
                line=dict(color="red"),
                name="shortest path",
            )
        )

    for p in points.keys():
        fig.add_trace(
            go.Scatter3d(
                x=[points[p]["p"][0]],
                y=[points[p]["p"][1]],
                z=[points[p]["p"][2]],
                mode="markers",
                marker=dict(color=[points[p]["color"]], size=4),
                name=p,
            )
        )

    fig.update_layout(scene=dict(zaxis=dict(range=[min_xyz - 1, max_xyz + 1])))
    fig.write_html(out_filename)
