import pyodbc
import os

def get_conn():
    return pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER={os.environ.get("DB_SERVER")};'
        f'DATABASE={os.environ.get("DB_NAME")};'
        f'UID={os.environ.get("DB_USER")};'           
        f'PWD={os.environ.get("DB_PASSWORD")};'      
        'TrustServerCertificate=yes;'
    )