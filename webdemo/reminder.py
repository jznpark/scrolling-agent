"""Opt-in break reminders for the demo; no changes to the model or its RNG."""

from dataclasses import dataclass, fields
import math
from typing import Any


@dataclass(frozen=True)
class ReminderSettings:
    enabled: bool = False
    after_seconds: float = 30.0
    snooze_seconds: float = 15.0

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any]) -> "ReminderSettings":
        values = {}
        names = {f.name for f in fields(cls)}
        for key, value in overrides.items():
            if not key.startswith("reminder."):
                continue
            name = key.removeprefix("reminder.")
            if name not in names:
                raise ValueError(f"unknown setting {key!r}")
            if name == "enabled":
                if not isinstance(value, bool):
                    raise ValueError("reminder.enabled must be true or false")
            else:
                if isinstance(value, bool):
                    raise ValueError(f"{key} must be a positive number of seconds")
                value = float(value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{key} must be a positive finite number of seconds")
            values[name] = value
        return cls(**values)


class BreakReminder:
    def __init__(self, settings: ReminderSettings, dt: float):
        self.settings = settings
        self.dt = dt
        self.next_step = self._steps(settings.after_seconds)
        self.pending = False
        self.stopped = False
        self.count = 0

    def _steps(self, seconds: float) -> int:
        # Round up to the next simulation step; never notify before the threshold.
        return max(1, math.ceil(seconds / self.dt - 1e-10))

    def check(self, step: int, *, finished: bool) -> bool:
        if (self.settings.enabled and not finished and not self.stopped
                and not self.pending and step >= self.next_step):
            self.pending = True
            self.count += 1
            return True
        return False

    def respond(self, choice: str, step: int) -> None:
        if choice not in ("continue", "stop"):
            raise ValueError("reminder choice must be 'continue' or 'stop'")
        if not self.pending or self.stopped:
            raise ValueError("no break reminder is waiting for a response")
        self.pending = False
        if choice == "stop":
            self.stopped = True
        else:
            self.next_step = step + self._steps(self.settings.snooze_seconds)

    def state(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "after_seconds": self.settings.after_seconds,
            "snooze_seconds": self.settings.snooze_seconds,
            "pending": self.pending,
            "stopped": self.stopped,
            "count": self.count,
            "next_at": round(self.next_step * self.dt, 6)
            if self.settings.enabled and not self.stopped else None,
        }
