import numpy as np


class APFeatureExtractor:
    def __init__(self, time, voltage):
        self.time = np.array(time)
        self.voltage = np.array(voltage)

    def extract_features(self):

        # Resting membrane potential and peak
        rmp = float(np.min(self.voltage))
        peak = float(np.max(self.voltage))

        amplitude = peak - rmp

        if amplitude < 20:
            raise ValueError("No valid action potential detected.")

        peak_idx = np.argmax(self.voltage)

        repol_voltage = self.voltage[peak_idx:]
        repol_time = self.time[peak_idx:]

        apd50_level = peak - 0.5 * amplitude
        apd90_level = peak - 0.9 * amplitude

        def interpolate_crossing(voltage, time, level):
            crossings = np.where(voltage <= level)[0]

            if len(crossings) == 0:
                raise ValueError("Repolarization level not reached.")

            idx = crossings[0]

            if idx == 0:
                return time[0]

            v1 = voltage[idx - 1]
            v2 = voltage[idx]

            t1 = time[idx - 1]
            t2 = time[idx]

            if v1 == v2:
                return t2

            fraction = (level - v1) / (v2 - v1)

            return t1 + fraction * (t2 - t1)


        apd50_time = interpolate_crossing(
            repol_voltage,
            repol_time,
            apd50_level,
        )

        apd90_time = interpolate_crossing(
            repol_voltage,
            repol_time,
            apd90_level,
        )

        apd50 = float(apd50_time - self.time[peak_idx])
        apd90 = float(apd90_time - self.time[peak_idx])

        triangulation = apd90 - apd50

        if triangulation < 0:
            raise ValueError("Invalid APD ordering.")

        return {
            "RMP": rmp,
            "Peak": peak,
            "APD50": apd50,
            "APD90": apd90,
            "Triangulation": triangulation,
        }