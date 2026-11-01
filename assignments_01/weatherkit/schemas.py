"""Validate external weather JSON before processing it."""

from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Parallel hourly timestamps, Celsius temperatures, and millimeter rainfall."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def matching_lengths(self) -> "HourlyBlock":
        if not (len(self.time) == len(self.temperature_2m) == len(self.precipitation)):
            raise ValueError("Hourly lists must have matching lengths")
        return self


class WeatherResponse(BaseModel):
    """Weather observations for a location with valid geographic coordinates."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock
