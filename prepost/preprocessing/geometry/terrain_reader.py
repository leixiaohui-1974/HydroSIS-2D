"""
Terrain data reader for HydroSIS-2D

Read and import terrain elevation data from various formats.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class TerrainData:
    """
    Container for terrain elevation data

    Attributes:
        data: 2D array of elevation values
        ncols: Number of columns
        nrows: Number of rows
        xllcorner: X coordinate of lower-left corner
        yllcorner: Y coordinate of lower-left corner
        cellsize: Grid cell size
        nodata_value: Value representing no data
    """

    def __init__(self, data: np.ndarray, ncols: int, nrows: int,
                 xllcorner: float, yllcorner: float, cellsize: float,
                 nodata_value: float = -9999):
        """
        Initialize terrain data

        Args:
            data: 2D elevation array
            ncols: Number of columns
            nrows: Number of rows
            xllcorner: X coordinate of lower-left corner
            yllcorner: Y coordinate of lower-left corner
            cellsize: Cell size
            nodata_value: No data value
        """
        self.data = data
        self.ncols = ncols
        self.nrows = nrows
        self.xllcorner = xllcorner
        self.yllcorner = yllcorner
        self.cellsize = cellsize
        self.nodata_value = nodata_value

    @property
    def xmax(self) -> float:
        """Maximum X coordinate"""
        return self.xllcorner + self.ncols * self.cellsize

    @property
    def ymax(self) -> float:
        """Maximum Y coordinate"""
        return self.yllcorner + self.nrows * self.cellsize

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax)"""
        return (self.xllcorner, self.xmax, self.yllcorner, self.ymax)

    def get_statistics(self) -> Dict[str, float]:
        """Compute terrain statistics"""
        # Mask out nodata values
        valid_data = self.data[self.data != self.nodata_value]

        if len(valid_data) == 0:
            return {}

        return {
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'median': float(np.median(valid_data))
        }


class ASCIIGridReader:
    """
    Read ArcGIS ASCII Grid format terrain data

    Format:
        ncols         100
        nrows         50
        xllcorner     0.0
        yllcorner     0.0
        cellsize      2.0
        NODATA_value  -9999
        elevation_data...
    """

    @staticmethod
    def read(filepath: str) -> TerrainData:
        """
        Read ASCII Grid file

        Args:
            filepath: Path to .asc file

        Returns:
            TerrainData object
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            # Read header
            header = {}
            for i in range(6):
                line = f.readline().strip().split()
                if len(line) >= 2:
                    key = line[0].lower()
                    value = float(line[1]) if '.' in line[1] else int(line[1])
                    header[key] = value

            # Read elevation data
            data = []
            for line in f:
                row = [float(x) for x in line.split()]
                if row:
                    data.append(row)

        # Convert to numpy array
        data_array = np.array(data)

        # Create TerrainData object
        terrain = TerrainData(
            data=data_array,
            ncols=header.get('ncols', data_array.shape[1]),
            nrows=header.get('nrows', data_array.shape[0]),
            xllcorner=header.get('xllcorner', 0.0),
            yllcorner=header.get('yllcorner', 0.0),
            cellsize=header.get('cellsize', 1.0),
            nodata_value=header.get('nodata_value', -9999)
        )

        logger.info(f"Loaded ASCII Grid: {filepath}")
        logger.info(f"  Size: {terrain.nrows} × {terrain.ncols}")
        logger.info(f"  Extent: {terrain.extent}")
        logger.info(f"  Cell size: {terrain.cellsize}")

        return terrain

    @staticmethod
    def write(terrain: TerrainData, filepath: str) -> None:
        """
        Write terrain data to ASCII Grid format

        Args:
            terrain: TerrainData object
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            # Write header
            f.write(f"ncols         {terrain.ncols}\n")
            f.write(f"nrows         {terrain.nrows}\n")
            f.write(f"xllcorner     {terrain.xllcorner}\n")
            f.write(f"yllcorner     {terrain.yllcorner}\n")
            f.write(f"cellsize      {terrain.cellsize}\n")
            f.write(f"NODATA_value  {terrain.nodata_value}\n")

            # Write data
            for row in terrain.data:
                f.write(" ".join([str(x) for x in row]) + "\n")

        logger.info(f"Wrote ASCII Grid: {filepath}")


class TerrainReader:
    """
    Main terrain reader interface

    Supports multiple formats and provides convenience methods.
    """

    @staticmethod
    def read_ascii_grid(filepath: str) -> TerrainData:
        """Read ASCII Grid format"""
        return ASCIIGridReader.read(filepath)

    @staticmethod
    def from_array(data: np.ndarray,
                  xllcorner: float = 0.0,
                  yllcorner: float = 0.0,
                  cellsize: float = 1.0) -> TerrainData:
        """
        Create TerrainData from numpy array

        Args:
            data: 2D elevation array with shape (ncols, nrows)
            xllcorner: X coordinate of lower-left corner
            yllcorner: Y coordinate of lower-left corner
            cellsize: Cell size

        Returns:
            TerrainData object
        """
        ncols, nrows = data.shape

        return TerrainData(
            data=data,
            ncols=ncols,
            nrows=nrows,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
            cellsize=cellsize
        )


# Convenience functions

def load_terrain(filepath: str) -> TerrainData:
    """
    Load terrain from file (auto-detect format)

    Args:
        filepath: Path to terrain file

    Returns:
        TerrainData object
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.asc', '.txt']:
        return ASCIIGridReader.read(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_terrain(terrain: TerrainData, filepath: str) -> None:
    """
    Save terrain to file

    Args:
        terrain: TerrainData object
        filepath: Output file path
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.asc', '.txt']:
        ASCIIGridReader.write(terrain, filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
