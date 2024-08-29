from pathlib import Path
from typing import Dict, List, Tuple
import json

from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from llama_index.core import SQLDatabase

Base = declarative_base()


class Room(Base):
    """
    SQLAlchemy model for the rooms table, with information on their centers and dimensions.
    """

    __tablename__ = "rooms"
    room = Column(String(16), primary_key=True)
    center_x = Column(Float)
    center_y = Column(Float)
    center_z = Column(Float)
    size_x = Column(Float)
    size_y = Column(Float)
    size_z = Column(Float)

    _column_to_index: str = "room"


class DetectedObject(Base):
    """
    SQLAlchemy model for the detected_objects table, with information on their
    detected objects in the room in terms of their classes, positions, sizes
    and corresponding room.
    """

    __tablename__ = "detected_objects"
    id = Column(Integer, primary_key=True)
    class_name = Column(String(16), nullable=False)
    room = Column(String(16), ForeignKey("rooms.room"), nullable=False)
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    size_x = Column(Float)
    size_y = Column(Float)
    size_z = Column(Float)

    _column_to_index: str = "class_name"


def parse_data(data: dict) -> Tuple[List[dict], List[dict]]:
    """
    Parses raw JSON data into structured room- and object-related data.

    Args:
        data (dict): raw JSON data containing room and object information

    Returns:
        Tuple[List[dict], List[dict]]: structured room and object data
    """
    rows_rooms = []
    rows_objects = []
    for room, details in data.items():
        room_data = details["room"]
        rows_rooms.append(
            {
                "room": room,
                "center_x": room_data["center"][0],
                "center_y": room_data["center"][1],
                "center_z": room_data["center"][2],
                "size_x": room_data["sizes"][0],
                "size_y": room_data["sizes"][1],
                "size_z": room_data["sizes"][2],
            }
        )
        for ind, obj in details["objects"].items():
            rows_objects.append(
                {
                    "id": int(ind),
                    "class_name": obj["class_name"],
                    "room": room,
                    "position_x": obj["center"][0],
                    "position_y": obj["center"][1],
                    "position_z": obj["center"][2],
                    "size_x": obj["sizes"][0],
                    "size_y": obj["sizes"][1],
                    "size_z": obj["sizes"][2],
                }
            )
    return rows_rooms, rows_objects


def load_sql_database(json_file_path: Path) -> SQLDatabase:
    """
    Loads a SQL database from a JSON file containing room and object information.

    Args:
        json_file_path (Path): path to the JSON file containing object detections and room information

    Returns:
        SQLDatabase: SQL database with room- and object-related tables
    """
    f = open(json_file_path)
    data = json.load(f)

    rows_rooms, rows_objects = parse_data(data)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        session.add_all([Room(**room) for room in rows_rooms])
        session.add_all([DetectedObject(**obj) for obj in rows_objects])
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

    return SQLDatabase(
        engine, include_tables=[Room.__tablename__, DetectedObject.__tablename__]
    )


def get_dict_to_index() -> Tuple[str, str]:
    """
    Returns the dictionary mapping table names to the column whose entries
    need to be indexed and used in RAG.

    Returns:
        Tuple[str, str]: dictionary mapping table names to the to-be-indexed columns
    """
    tables = [Room, DetectedObject]
    to_index: Dict[str, str] = {}
    for table in tables:
        to_index[table.__tablename__] = table._column_to_index
    return to_index
