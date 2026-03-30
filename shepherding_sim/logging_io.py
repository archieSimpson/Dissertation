import csv
import os


class CSVLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.fieldnames = None

        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def write_row(self, row: dict):
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class SheepStateLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header_written = False

        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def write_states(self, time_value: float, sheep_list):
        fieldnames = [
            "time", "sheep_id", "x", "y", "vx", "vy", "speed",
            "alert", "state", "fear_strength", "max_speed_multiplier",
            "reaction_delay_frames", "graze_bias"
        ]

        mode = "a" if self.header_written else "w"
        with open(self.filepath, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not self.header_written:
                writer.writeheader()
                self.header_written = True

            for sh in sheep_list:
                writer.writerow({
                    "time": time_value,
                    "sheep_id": sh.idx,
                    "x": float(sh.pos[0]),
                    "y": float(sh.pos[1]),
                    "vx": float(sh.vel[0]),
                    "vy": float(sh.vel[1]),
                    "speed": float((sh.vel[0] ** 2 + sh.vel[1] ** 2) ** 0.5),
                    "alert": int(sh.alert),
                    "state": sh.state,
                    "fear_strength": sh.fear_strength,
                    "max_speed_multiplier": sh.max_speed_multiplier,
                    "reaction_delay_frames": sh.reaction_delay_frames,
                    "graze_bias": sh.graze_bias,
                })