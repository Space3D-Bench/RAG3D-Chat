# SQL module
SQL_FUN_PROMPT = "Answers natural language questions regarding quantitative information about the apartment, such as the categories and number of objects in rooms, their 3D positions, sizes, and the rooms they are associated with. It is the exclusive source for 3D positional data of objects within the apartment. It does not contain data about the visual appearance of the objects or rooms, nor the spatial relations between objects (like A is next to B, A is under B). It can create SQL queries involving the calculation of Euclidean distances between objects."
SQL_IN_PROMPT = "Natural language query (NOT an SQL query)."
SQL_FUN_DIST_PROMPT = "Returns the data related to Euclidean distance between multiple objects, provided 3D position of an object which we want to compare with others is in the query. It can e.g. get the names of the objects closest/furthest/within certain distance from the given 3D position."
SQL_IN_DIST_PROMPT = "Natural language query containing the 3D position of an object to compare with others and the description of the task objective."
SQL_OUT_PROMPT = "Natural language answer to the query."
SQL_TABLE_CONTEXTS = {
    "rooms": "The table contains the information on rooms, their centers position and the dimensions.",
    "detected_objects": "The table contains all detected objects in the apartment, with their corresponding room association, position, sizes and class name.",
}
SQL_MENTIONED_ELEMENTS_PROMPT = ('Given an input query, return all the names of the objects and rooms mentioned or implied in a form of string in a JSON format with keys "detected_objects" and "rooms", and values being a list with strings with the names:\n "rooms": [name-of-the-room1, name-of-the-room2], "detected_objects": [name-of-the-object1, name-of-the-object2]\n'
    'Example: for query=Is there an object you can sit on in the living room?, return the following dictionary: "rooms": ["living room"], "detected_objects": ["object you can sit on"].')
SQL_DIST_DISCARD_PROMPT = "Discard the object which is within 0.05 meters with respect to the one in query, since we don't want to compare the object to itself."

# Navigation module
NAV_FUN_ACTUAL_PROMPT = "Calculates the distance between two points, considering obstacles and non-navigable areas, when provided with the 3D positions of those points. It does not independently determine the positions of objects or points in space, the positions need to be included in the input query. It is the default distance measurement when the query implies walking or getting from one place to another, since it considers walls separating the rooms."
NAV_OUT_ACTUAL_PROMPT = "Information on the actual distance between the points (considering the obstacles and non-navigable areas). If the input was in correct format, it can still happen that the path between objects is not navigable."
NAV_IN_PROMPT = "Natural language query specifying the positions of the start and end, including X, Y and Z components."
NAV_FUN_LINE_PROMPT = "Having the description of the 3D positions of start and goal, returns the distance between them in straight line, NOT considering any obstacles. It does not independently determine the positions of objects or points in space, the positions need to be included in the input query."
NAV_OUT_LINE_PROMPT = "Information on the straight-line distance between the points (NOT considering the obstacles)."

NAV_SYSTEM_PROMPT = (
    "Act as a text-to-positions converter. Your job is to process a natural language query which contains information about 3D positions of two objects. "
    "Return an answer in the following format: '(x1,y1,z1),(x2,y2,z2)', where x1,y1,z1 correspond to 3D position of the first object and x2,y2,z2 are the coordinates of the second object. "
    "Make sure that the xyz positions have no more than three decimal values. "
    "If in the input there are no positions given, the positions do not have all 3 components each, the values cannot be parsed to values or if there are too few/too many objects, return 'None'. "
    "Do not perform actions that are not related to this task."
)

# Text document module
TEXT_FUN_PROMPT = "If the query concerns two rooms or more, it can answer questions regarding visual data about the scenes, such as appearances of rooms and objects, and about spatial relations between the objects (such as A is on B, A is next to B etc.). It does not provide any quantitative or positional data, such as the 3D positions of objects."
TEXT_IN_PROMPT = "Query asking for a visual data of an objects, rooms or scenes."
TEXT_OUT_PROMPT = "String being the visual data of the objects, rooms or scenes."
TEXT_METADATA_PROMPT = "Your job is to decide whether the given query concerns a specific room. If yes, return ONLY the name of the room from the pool {}. If not, or if the room name is not close to any of the available ones, return 'None'."

# Image document module
IMAGE_FUN_PROMPT = "If the query concerns exactly one room, it can answer questions regarding the spatial relationships between objects within this room (such as A is on B, A is under B), count the number of visible objects or describe the surroundings in terms of visual appearance. It does not provide any 3D positions of objects nor the cumulative data about more than one room."
IMAGE_IN_PROMPT = "Query asking for a visual data of an object, room or scene."
IMAGE_OUT_PROMPT = "String being the visual data of the object, room or scene."
IMAGE_METADATA_PROMPT = "Your job is to decide whether the given query concerns a specific room. If yes, return ONLY the name of the room from the pool {}. If not, or if the room name is not close to any of the available ones, return 'None'."
