import numpy as np
import ising.core as isc


class IsingModel:
    @staticmethod
    def __init_system(n):
        return np.array(np.random.choice([-1, 1], size=(n, n)), dtype=np.int8)

    def __init__(self, size: int, init_tempetature: float):
        self.system = IsingModel.__init_system(size)
        self.temperature = init_tempetature

    @property
    def temperature(self) -> float:
        return self.__temperature

    @temperature.setter
    def temperature(self, value: float):
        self.__temperature = float(np.clip(value, 0, 100))

    @property
    def internal_energy(self):
        return isc.calc_internal_energy(self.system)

    @property
    def heat_capacity(self) -> float:
        return isc.calc_heat_capacity(self.system)

    @property
    def magnetization(self) -> float:
        return isc.calc_magnetization(self.system)

    def restart(self):
        self.system = IsingModel.__init_system(self.system.shape[0])

    def step(self, epochs: int):
        self.system = isc.update_cells(self.system, epochs, self.temperature)
