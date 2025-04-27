# db.py
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, Boolean, TIMESTAMP
from sqlalchemy.sql import func
from databases import Database
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
database = Database(DATABASE_URL)
metadata = MetaData()

violations_table = Table(
    "violations",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("plate_number", String(50)),
    Column("violation_type", String(100)),
    Column("detected_at", TIMESTAMP, server_default=func.now()),
    Column("image_url", Text),
    Column("is_notified", Boolean, default=False),
)

async def save_violation_db(plate_number: str, violation_type: str, image_url: str):
    query = violations_table.insert().values(
        plate_number=plate_number,
        violation_type=violation_type,
        image_url=image_url,
        is_notified=False
    )
    await database.execute(query)
