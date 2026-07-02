import numpy as np


class APFeatureExtractor:
    def __init__(self, time, voltage):
        self.time = np.array(time)
        self.voltage = np.array(voltage)

    def extract_features(self):
        rmp = np.min(self.voltage)
        peak = np.max(self.voltage)

        amplitude = peak - rmp

        apd90_level = peak - 0.9 * amplitude
        apd50_level = peak - 0.5 * amplitude

        peak_idx = np.argmax(self.voltage)

        repol_voltage = self.voltage[peak_idx:]
        repol_time = self.time[peak_idx:]

        apd90_idx = np.argmin(np.abs(repol_voltage - apd90_level))
        apd50_idx = np.argmin(np.abs(repol_voltage - apd50_level))

        apd90 = repol_time[apd90_idx] - self.time[peak_idx]
        apd50 = repol_time[apd50_idx] - self.time[peak_idx]

        triangulation = apd90 - apd50

        return {
            "RMP": float(rmp),
            "Peak": float(peak),
            "APD50": float(apd50),
            "APD90": float(apd90),
            "Triangulation": float(triangulation),
        }