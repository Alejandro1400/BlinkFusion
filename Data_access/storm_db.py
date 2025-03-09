import sqlite3
import os


class STORMDatabaseManager:
    """Handles database connections and operations for the STORM database."""

    def __init__(self, database_folder, db_name="storm.db"):
        """
        Initialize the STORMDatabaseManager with a base folder and database name.

        Args:
            database_folder (str): The path to the folder where the database will be stored.
            db_name (str): The name of the database file (default: 'storm.db').
        """
        self.db_path = os.path.join(database_folder, db_name)
        self.conn = self.connect()
        self.cursor = self.conn.cursor()

    def connect(self):
        """Establish and return a SQLite connection."""
        return sqlite3.connect(self.db_path)

    def initialize_database(self):
        """Initialize tables for the STORM database."""
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

        database_metadata = {}
        for name, value, type_ in metadata_entries:
            if name not in database_metadata:
                database_metadata[name] = []
            database_metadata[name].append((value, type_))

        return database_metadata

    def storm_folders_without_localizations(self):
        """
        Retrieves folders where TIFF files exist but have no localization data.

        Returns:
            list: A list of unique folder paths that contain TIFF files but no localization data.
        """
        self.cursor.execute("""
            SELECT DISTINCT m.id AS metadata_id, m.file_name AS folder_path
            FROM metadata m
            LEFT JOIN localizations l ON m.id = l.metadata_id
            WHERE m.name = 'Date' AND l.metadata_id IS NULL
        """)
        
        return self.cursor.fetchall()  # Returns a list of tuples: (metadata_id, folder_path)
    

    def save_molecules(self, molecules, metadata_id):
        """
        Saves molecules, tracks, and localizations to the database in an optimized manner.

        Args:
            molecules (list[Molecule]): List of Molecule objects.
            metadata_id (int): Metadata ID linking these entries.
        """

        # Prepare bulk insert lists
        molecule_data = []
        track_data = []
        localization_data = []

        for molecule in molecules:
            # 1️⃣ Molecule Data
            molecule_data.append((
                molecule.molecule_id, molecule.start_track, molecule.end_track,
                molecule.num_tracks, molecule.total_on_time, metadata_id
            ))

            for track in molecule.tracks:
                # 2️⃣ Track Data
                track_data.append((
                    track.track_id, track.x, track.y, 0,  # z is defaulted to 0
                    track.start_frame, track.end_frame, track.intensity,
                    track.offset, track.bkgstd, track.uncertainty,
                    len(track.gaps), track.on_time, track.off_time,
                    molecule.molecule_id, metadata_id
                ))

                for loc in track.localizations:
                    # 3️⃣ Localization Data
                    localization_data.append((
                        loc.id, loc.frame, loc.x, loc.y, loc.sigma, loc.intensity,
                        loc.offset, loc.bkgstd, loc.uncertainty, track.track_id, metadata_id
                    ))

        # Execute batch insertions
        try:
            self.cursor.executemany("""
                INSERT INTO molecules (molecule_id, start_track, end_track, num_tracks, total_on_time, metadata_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, molecule_data)

            self.cursor.executemany("""
                INSERT INTO tracks (id, x, y, z, start_frame, end_frame, intensity, offset, bkgstd, uncertainty,
                                   gaps, on_time, off_time, molecule_id, metadata_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, track_data)

            self.cursor.executemany("""
                INSERT INTO localizations (id, frame, x, y, sigma, intensity, offset, bkgstd, uncertainty, track_id, metadata_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, localization_data)

            self.conn.commit()
            print(f"✅ Successfully inserted {len(molecule_data)} molecules, {len(track_data)} tracks, and {len(localization_data)} localizations.")

        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"❌ Database error: {e}")

    def save_time_series(self, time_series_df, metadata_id, interval_frames):
        """
        Saves time series data to the database.

        Args:
            time_series_df (pd.DataFrame): DataFrame containing time-series metrics.
            metadata_id (int): ID referencing the metadata record.
        """
        # Convert DataFrame to list of tuples for bulk insertion
        time_series_data = [
            (row['Duty Cycle'], row['Survival Fraction'], row['Population Mol'],
            row['SC per Mol'], row['On Time per SC (s)'], int(index), int(index + interval_frames), metadata_id)
            for index, row in time_series_df.iterrows()
        ]

        # Bulk insert into database
        self.cursor.executemany("""
            INSERT INTO time_series (duty_cycle, survival_fraction, population_mol, 
                                    sc_per_mol, on_time_per_sc, start_frame, end_frame, metadata_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, time_series_data)

        self.conn.commit()
        print(f"✅ Successfully inserted {len(time_series_data)} time series records.")

    def get_experiment_settings(self, metadata_id):
        """
        Retrieves total frames and exposure time from metadata where the root tag is 'czi pulsestorm'.

        Args:
            metadata_id (int): The metadata ID to query.

        Returns:
            tuple: (total_frames, exposure_time) or (None, None) if not found.
        """
        self.cursor.execute("""
            SELECT name, value FROM metadata
            WHERE metadata_id = ? AND tag = 'czi-pulsestorm' AND name IN ('FRAMES', 'EXPOSURE')
            """, (metadata_id,))
        
        # Fetch and store results in a dictionary
        metadata_values = {row[0]: float(row[1]) for row in self.cursor.fetchall()}
        
        total_frames = metadata_values.get('FRAMES', None)
        exposure_time = metadata_values.get('EXPOSURE', None)

        return total_frames, exposure_time

    def close(self):
        """Close the database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Define the database folder path
    storm_folder = "C://Users//usuario//Box//For Alejandro//STORM Data"

    # Initialize STORM database
    storm_db = STORMDatabaseManager(storm_folder)
    storm_db.initialize_database()
    storm_db.close()

    print("STORM database initialized successfully.")
