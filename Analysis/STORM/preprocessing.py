import pandas as pd

class Preprocessor:
    """
    Handles preprocessing of localization data for different file types.
    """

    def __init__(self, df: pd.DataFrame, file_type: str):
        """
        Initializes the preprocessor.

        Parameters:
        - df (pd.DataFrame): Raw input DataFrame.
        - file_type (str): Type of input file ('trackmate' or 'thunderstorm').
        """
        self.df = df
        self.file_type = file_type

    def prepare_columns(self) -> pd.DataFrame:
        """
        Standardizes columns for 'trackmate' files.

        Returns:
        - pd.DataFrame: Processed DataFrame with standardized columns.
        """
        df = self.df.copy()

        if self.file_type == "trackmate":
            first_valid_index = df[df["LABEL"].astype(str).str.startswith("ID")].index.min()
            df = df.loc[first_valid_index:]

            df.columns = df.columns.str.upper()
            df = df[
                ["ID", "FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z", "QUALITY", "TOTAL_INTENSITY_CH1", "SNR_CH1", "TRACK_ID"]
            ]
            df.columns = ["ID", "FRAME", "X", "Y", "Z", "QUALITY", "INTENSITY", "SNR", "TRACK_ID"]

            df["ID"] = df["ID"].astype(int)
            df["TRACK_ID"] = df["TRACK_ID"].fillna(0).astype(int)
            df["X"] = df["X"].astype(float)
            df["Y"] = df["Y"].astype(float)
            df["Z"] = df["Z"].astype(float)
            df["FRAME"] = df["FRAME"].astype(int)
            df["QUALITY"] = df["QUALITY"].astype(float)
            df["INTENSITY"] = df["INTENSITY"].astype(float)
            df["SNR"] = df["SNR"].astype(float)

        return df

    def create_tracking_events(self) -> pd.DataFrame:
        """
        Creates tracking events from the preprocessed DataFrame.

        Returns:
        - pd.DataFrame: Tracking events with weighted coordinates and statistical properties.
        """
        df = self.prepare_columns()
        weight_column = "UNCERTAINTY" if self.file_type == "thunderstorm" else "QUALITY"

        if weight_column not in df.columns:
            df[weight_column] = 1

        # Remove TRACK_ID = 0
        filtered_locs = df[df["TRACK_ID"] != 0].copy()

        filtered_locs["weighted_x"] = filtered_locs["X"] * filtered_locs[weight_column]
        filtered_locs["weighted_y"] = filtered_locs["Y"] * filtered_locs[weight_column]
        filtered_locs["weighted_z"] = filtered_locs["Z"] * filtered_locs[weight_column]

        grouped = filtered_locs.groupby("TRACK_ID")
        result = grouped.apply(lambda g: pd.Series({
            "X": g["weighted_x"].sum() / g[weight_column].sum(),
            "Y": g["weighted_y"].sum() / g[weight_column].sum(),
            "Z": g["weighted_z"].sum() / g[weight_column].sum(),
            "QUALITY": g["QUALITY"].sum(),
            "START_FRAME": g["FRAME"].min(),
            "END_FRAME": g["FRAME"].max(),
            "GAPS": list(set(range(g["FRAME"].min(), g["FRAME"].max() + 1)) - set(g["FRAME"])),
        })).reset_index()

        return result
