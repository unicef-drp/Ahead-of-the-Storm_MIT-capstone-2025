import abc
import geopandas as gpd


class ExposureLayer(abc.ABC):
    """Abstract base class for hazard exposure layers (e.g., hurricane, flood)."""

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def compute_grid(self) -> gpd.GeoDataFrame:
        """Compute the exposure probability grid (0-1 per cell)."""
        pass

    @abc.abstractmethod
    def plot(self, ax=None):
        """Plot the exposure layer."""
        pass


class VulnerabilityLayer(abc.ABC):
    """Abstract base class for vulnerability layers (e.g., schools, hospitals, population)."""

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def compute_grid(self) -> gpd.GeoDataFrame:
        """Compute the vulnerability grid (entity count/density per cell)."""
        pass

    @abc.abstractmethod
    def plot(self, ax=None):
        """Plot the vulnerability layer."""
        pass


class ImpactLayer(abc.ABC):
    """Abstract base class for impact layers (combining exposure and vulnerability)."""

    def __init__(
        self,
        exposure_layer: ExposureLayer,
        vulnerability_layer: VulnerabilityLayer,
        config,
    ):
        self.exposure_layer = exposure_layer
        self.vulnerability_layer = vulnerability_layer
        self.config = config

    @abc.abstractmethod
    def compute_impact(self) -> gpd.GeoDataFrame:
        """Compute the impact grid (e.g., expected affected entities per cell)."""
        pass

    @abc.abstractmethod
    def plot(self, ax=None):
        """Plot the impact layer."""
        pass

    @abc.abstractmethod
    def expected_impact(self) -> float:
        """Compute the expected number of affected entities (sum over grid)."""
        pass

    @abc.abstractmethod
    def best_case(self) -> float:
        """Compute the best-case (minimum) number of affected entities."""
        pass

    @abc.abstractmethod
    def worst_case(self) -> float:
        """Compute the worst-case (maximum) number of affected entities."""
        pass
