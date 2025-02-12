import sqlite3
import os

# Paths for SQLite databases (ensure they sync with Box)
BOX_PATH = "/Users/YourName/Box"  # Update with actual Box Drive path
FILAMENTS_DB_PATH = os.path.join(BOX_PATH, "filaments.db")
STORM_DB_PATH = os.path.join(BOX_PATH, "storm.db")

def connect(db_path):
    """Establish a connection to a SQLite database."""
    return sqlite3.connect(db_path)

def initialize_filaments_db():
    """Creates tables for the Filaments database."""
    conn = connect(FILAMENTS_DB_PATH)
    cursor = conn.cursor()

    # Metadata Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            value TEXT
        )
    """)

    # Ridge Metrics Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ridge_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number_of_ridges INTEGER,
            ridge_junction_ratio REAL,
            mean_length REAL,
            cv_length REAL,
            cv_width REAL,
            mean_intensity REAL,
            ROI TEXT,
            metadata_id INTEGER,
            FOREIGN KEY(metadata_id) REFERENCES metadata(id)
        )
    """)

    # SOAC Results Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS soac_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file TEXT,
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
        )
    """)

    conn.commit()
    conn.close()


def initialize_storm_db():
    """Creates tables for the STORM database."""
    conn = connect(STORM_DB_PATH)
    cursor = conn.cursor()

    # Metadata Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            value TEXT
        )
    """)

    # Localizations Table
    cursor.execute("""
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
            FOREIGN KEY(track_id) REFERENCES tracks(id),
            metadata_id INTEGER,
            FOREIGN KEY(metadata_id) REFERENCES metadata(id)
        )
    """)

    # Tracks Table
    cursor.execute("""
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
            FOREIGN KEY(molecule_id) REFERENCES molecules(molecule_id),
            metadata_id INTEGER,
            FOREIGN KEY(metadata_id) REFERENCES metadata(id)
        )
    """)

    # Molecules Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS molecules (
            molecule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_track INTEGER,
            end_track INTEGER,
            num_tracks INTEGER,
            total_on_time REAL,
            metadata_id INTEGER,
            FOREIGN KEY(metadata_id) REFERENCES metadata(id)
        )
    """)

    # Time Series Table
    cursor.execute("""
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
        )
    """)

    conn.commit()
    conn.close()

def initialize_databases():
    """Initialize both Filaments and STORM databases."""
    initialize_filaments_db()
    initialize_storm_db()
    print("Databases initialized successfully.")

if __name__ == "__main__":
    initialize_databases()