import sqlite3
import os


class FilamentsDatabaseManager:
    """Handles database connections and operations for the Filaments database."""

    def __init__(self, database_folder, db_name="filaments.db"):
        """
        Initialize the FilamentsDatabaseManager with a base folder and database name.

        Args:
            database_folder (str): The path to the folder where the database will be stored.
            db_name (str): The name of the database file (default: 'filaments.db').
        """
        self.db_path = os.path.join(database_folder, db_name)
        self.conn = self.connect()
        self.cursor = self.conn.cursor()

    def connect(self):
        """Establish and return a SQLite connection."""
        return sqlite3.connect(self.db_path)

    def initialize_database(self):
        """Initialize tables for the Filaments database."""
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                date TEXT NOT NULL,
                tag TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS ridge_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                date TEXT NOT NULL,
                number_of_ridges INTEGER,
                ridge_junction_ratio REAL,
                mean_length REAL,
                cv_length REAL,
                cv_width REAL,
                mean_intensity REAL,
                ROI TEXT,
                metadata_id INTEGER,
                FOREIGN KEY(metadata_id) REFERENCES metadata(id)
            );

            CREATE TABLE IF NOT EXISTS soac_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                date TEXT NOT NULL,
                network INTEGER,
                snake INTEGER,
                point INTEGER,
                junction INTEGER,
                x REAL,
                y REAL,
                z REAL,
                length REAL,
                sinuosity REAL,
                intensity REAL,
                background REAL,
                snr REAL,
                gaps INTEGER,
                metadata_id INTEGER,
                FOREIGN KEY(metadata_id) REFERENCES metadata(id)
            );
        """)
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Define the database folder path
    filament_folder = "C://Users//usuario//Box//For Alejandro//SOAC Filament Data"

    # Initialize Filaments database
    filaments_db = FilamentsDatabaseManager(filament_folder)
    filaments_db.initialize_database()
    filaments_db.close()

    print("Filaments database initialized successfully.")
