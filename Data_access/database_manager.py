import sqlite3
import os


class DatabaseManager:
    """Handles database connections and operations for STORM and Filaments databases."""

    def __init__(self, database_folder, db_name):
        """
        Initialize the DatabaseManager with a base folder and database name.

        Args:
            database_folder (str): The path to the folder where the database will be stored.
            db_name (str): The name of the database file (e.g., 'storm.db' or 'filaments.db').
        """
        self.db_path = os.path.join(database_folder, db_name)
        self.conn = self.connect()
        self.cursor = self.conn.cursor()
        self.db_name = db_name

    def connect(self):
        """Establish and return a SQLite connection."""
        return sqlite3.connect(self.db_path)

    def initialize_database(self):
        """Initialize database tables for either STORM or Filaments based on the database name."""
        if "filaments.db" in self.db_name:
            self._initialize_filaments_db()
        elif "storm.db" in self.db_name:
            self._initialize_storm_db()
        else:
            raise ValueError("Unknown database file.")

    def _initialize_filaments_db(self):
        """Creates tables for the Filaments database."""
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

    def _initialize_storm_db(self):
        """Creates tables for the STORM database."""
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

            CREATE TABLE IF NOT EXISTS localizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                sigma REAL NOT NULL,
                intensity REAL NOT NULL,
                offset REAL NOT NULL,
                bkgstd REAL NOT NULL,
                uncertainty REAL NOT NULL,
                track_id INTEGER,
                metadata_id INTEGER,
                FOREIGN KEY(track_id) REFERENCES tracks(id),
                FOREIGN KEY(metadata_id) REFERENCES metadata(id)
            );

            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL,
                y REAL,
                z REAL,
                start_frame INTEGER,
                end_frame INTEGER,
                intensity REAL,
                offset REAL,
                bkgstd REAL,
                uncertainty REAL,
                gaps INTEGER,
                on_time REAL,
                off_time REAL,
                molecule_id INTEGER,
                metadata_id INTEGER,
                FOREIGN KEY(molecule_id) REFERENCES molecules(molecule_id),
                FOREIGN KEY(metadata_id) REFERENCES metadata(id)
            );

            CREATE TABLE IF NOT EXISTS molecules (
                molecule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_track INTEGER,
                end_track INTEGER,
                num_tracks INTEGER,
                total_on_time REAL,
                metadata_id INTEGER,
                FOREIGN KEY(metadata_id) REFERENCES metadata(id)
            );

            CREATE TABLE IF NOT EXISTS time_series (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                duty_cycle REAL,
                survival_fraction REAL,
                population_mol INTEGER,
                sc_per_mol REAL,
                on_time_per_sc REAL,
                start_frame INTEGER,
                end_frame INTEGER,
                metadata_id INTEGER,
                FOREIGN KEY(metadata_id) REFERENCES metadata(id)
            );
        """)
        self.conn.commit()

    def save_metadata(self, metadata, file_name):
        """
        Save extracted metadata to the database, including the `tag` field.

        Args:
            metadata (list of dicts): List of metadata dictionaries with keys: id, type, value.
            file_name (str): The name of the file associated with the metadata.
        """
        date_value = next((entry['value'] for entry in metadata if entry['id'] == 'Date'), "Unknown")

        self.cursor.executemany("""
            INSERT INTO metadata (file_name, date, tag, name, type, value)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [(file_name, date_value, item['tag'], item['id'], item['type'], item['value']) for item in metadata])

        self.conn.commit()


    def load_storm_metadata(self):
        """
        Load distinct metadata from the database where `tag='PulseSTORM'`.

        Returns:
            dict: Metadata dictionary where keys are metadata names and values are lists of tuples (value, type).
        """
        self.cursor.execute("""
            SELECT DISTINCT name, value, type 
            FROM metadata 
            WHERE tag = 'pulsestorm'
            ORDER BY name, value;
        """)
        metadata_entries = self.cursor.fetchall()

        # Organize data in a dictionary
        database_metadata = {}
        for name, value, type_ in metadata_entries:
            if name not in database_metadata:
                database_metadata[name] = []
            database_metadata[name].append((value, type_))

        return database_metadata


    def close(self):
        """Close the database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Define the base database folder
    filament_folder = "C://Users//usuario//Box//For Alejandro//SOAC Filament Data" 
    storm_folder = "C://Users//usuario//Box//For Alejandro//STORM Data"  

    # Initialize Filaments database
    filaments_db = DatabaseManager(filament_folder, "filaments.db")
    filaments_db.initialize_database()
    filaments_db.close()

    # Initialize STORM database
    storm_db = DatabaseManager(storm_folder, "storm.db")
    storm_db.initialize_database()
    storm_db.close()

    print("Databases initialized successfully.")
