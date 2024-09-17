import logging
from typing import Annotated, Optional
from pathlib import Path
import numpy as np

import pygeodesic.geodesic as geodesic
import igl

from semantic_kernel.functions.kernel_function_decorator import kernel_function

from core.interfaces import AbstractLlmChat
from misc.navmesh_vis import visualize_navmesh_3d
from plugins.plugin_prompts import (
    NAV_FUN_ACTUAL_PROMPT,
    NAV_IN_PROMPT,
    NAV_OUT_ACTUAL_PROMPT,
    NAV_SYSTEM_PROMPT,
    NAV_OUT_LINE_PROMPT,
    NAV_OUT_LINE_PROMPT,
)

logger = logging.getLogger("NAV")


class NavPlugin:
    def __init__(
        self,
        navmesh_filepath: Path,
        llm_chat: AbstractLlmChat,
        vis_dirpath: Optional[Path] = None,
    ) -> None:
        """
        Constructor

        Args:
            navmesh_filepath (Path): path to the file with navigation mesh
                (compatible with Habitat Sim format)
            llm_chat (AbstractLlmChat): LLM chat used for positions extraction
            vis_dirpath (optional, Path): path to a directory in which the visualization
                of the resulting navigable paths should be stored

        Returns:
            None
        """
        self._vertices, self._faces = geodesic.read_mesh_from_file(navmesh_filepath)
        self._llm_chat: AbstractLlmChat = llm_chat
        self._vis_dirpath: Optional[Path] = vis_dirpath

    @kernel_function(description=NAV_FUN_ACTUAL_PROMPT, name="NavigationActual")
    def get_actual_distance_from_query(
        self, natural_language_descr: Annotated[str, NAV_IN_PROMPT]
    ) -> Annotated[str, NAV_OUT_ACTUAL_PROMPT]:
        """
        Calculates the real distance between the start and goal 3D points described by natural
        language, considering the obstacles.

        Args:
            natural_language_descr (str): natural language description of the 3D positions

        Returns:
            str: textual answer to the distance question between the points
        """

        logger.info("Calling the navigable distance function.")
        start, goal = self._nl_to_numpy(natural_language_descr)

        if isinstance(start, np.ndarray) and isinstance(goal, np.ndarray):
            response = self._get_navigable_distance(start=start, goal=goal)
        else:
            response = (
                "The query does not contain information about the objects' "
                "positions or the positions could not be parsed."
            )
        logger.info(f"Response: {response}")
        return response

    @kernel_function(description=NAV_OUT_LINE_PROMPT, name="NavigationLine")
    def get_straight_line_distance_from_query(
        self, natural_language_descr: Annotated[str, NAV_IN_PROMPT]
    ) -> Annotated[str, NAV_OUT_LINE_PROMPT]:
        """
        Calculates the straight-line distance between the start and goal 3D points described
        by natural language, considering the obstacles.

        Args:
            natural_language_descr (str): natural language description of the 3D positions

        Returns:
            str: textual answer to the straight-line distance question between the points
        """

        logger.info("Calling the straight-line distance function.")
        start, goal = self._nl_to_numpy(natural_language_descr)

        if isinstance(start, np.ndarray) and isinstance(goal, np.ndarray):
            dist = np.linalg.norm(goal - start)
            response = f"The distance between specified points is {dist} meters."
        else:
            response = (
                "The query does not contain information about the objects' "
                "positions or the positions could not be parsed."
            )
        logger.info(f"Response: {response}")
        return response

    def _nl_to_numpy(
        self, natural_language_descr: str
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Converts the natural language description of the 3D positions to the
        numpy array of coordinates.

        Args:
            natural_language_descr (str): natural language description of the 3D positions

        Returns:
            np.ndarray: 3D coordinates of the objects
        """
        logger.info(f"Natural language description: {natural_language_descr}")
        response = self._llm_chat.get_response(
            NAV_SYSTEM_PROMPT, natural_language_descr
        )
        logger.info(f"Retrieved positions: {response}")

        if response == "None":
            return None, None
        else:
            try:
                cleaned_string = response.replace("(", "").replace(")", "")
                split_values = cleaned_string.split(",")
                numpy_arrays = np.array(split_values, dtype=float).reshape(-1, 3)
                logger.info(
                    f"The distance is calculated between {numpy_arrays[0]} and {numpy_arrays[1]}."
                )
                return numpy_arrays[0], numpy_arrays[1]
            except Exception as _:
                logger.info("Could not parse the positions.")
                return None, None

    def _snap_to_closest_vertex(self, point):
        """
        Snaps the given point to the closest vertex in the navigation mesh.

        Args:
            point (np.ndarray[3,]): 3D position to snap to the closest vertex
        
        Returns:
            np.ndarray[3,]: closest point on the navigation mesh
            np.ndarray[3,]: closest vertex in the navigation mesh
            int: index of the closest vertex in the navigation mesh
        """
        _, _, closest_point = igl.signed_distance(np.array([point]), self._vertices, self._faces)
        distances = np.linalg.norm(self._vertices - closest_point, axis=1)
        closest_vertex_index = np.argmin(distances)
        closest_vertex = self._vertices[closest_vertex_index]
        return closest_point, closest_vertex, closest_vertex_index

    def _get_navigable_distance(self, start: np.ndarray, goal: np.ndarray) -> str:
        """
        Calculates the distance between the start and goal points, creates an answer
        based on the result. The start and goal points are given in the same coordinate
        system as the navigation mesh. If a visualization directory is provided, the
        visualization of the resulting path is saved there as an HTML file.

        Args:
            start (np.ndarray[3,]): starting 3D position, given in the Replica format,
                Z pointing up
            goal (np.ndarray[3,]): goal 3D position, given in the Replica format,
                Z pointing up

        Returns:
            str: textual answer to the distance question between the points
        """

        response = (
            "The path between specified objects is not navigable considering obstacles."
        )

        closest_s, closest_v_s, closest_vid_s = self._snap_to_closest_vertex(start)
        closest_g, closest_v_g, closest_vid_g = self._snap_to_closest_vertex(goal)
        geoalg = geodesic.PyGeodesicAlgorithmExact(self._vertices, self._faces)
        dist, path = geoalg.geodesicDistance(closest_vid_s, closest_vid_g)

        if path is not None and path.size != 0:
            response = f"The distance between specified points is {dist} meters."

        if self._vis_dirpath:
            points = {
                "start": {"p": start, "color": "blue"},
                "goal": {"p": goal, "color": "green"},
                "closest_start": {"p": closest_s, "color": "darkblue"},
                "closest_goal": {"p": closest_g, "color": "darkgreen"},
                "closest_v_start": {"p": closest_v_s, "color": "lightblue"},
                "closest_v_goal": {"p": closest_v_g, "color": "lightgreen"},
            }
            start_str = "-".join(map(str, np.round(start, 2)))
            goal_str = "-".join(map(str, np.round(goal, 2)))
            filename = self._vis_dirpath / f"{start_str}_{goal_str}.html"
            logger.info(f"Saving the visualization to {filename}")
            visualize_navmesh_3d(self._vertices, self._faces, filename, path, points)

        return response
