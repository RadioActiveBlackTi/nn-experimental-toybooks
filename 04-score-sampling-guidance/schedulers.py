class schedule_sigma:
    def __init__(self, sigma_min=0.01, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigma(self, t):
        return self.sigma_max * (self.sigma_min / self.sigma_max) ** t


class schedule_sigma_linear:
    def __init__(self, sigma_min=0.01, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigma(self, t):
        return self.sigma_max - t * (self.sigma_max - self.sigma_min)


class schedule_sigma_karras:
    def __init__(self, sigma_min=0.01, sigma_max=1.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigma(self, t):
        return (self.sigma_max ** (1 / self.rho) + t * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho


class scheduler_sigma_polynomial:
    def __init__(self, sigma_min=0.01, sigma_max=1.0, power=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.power = power

    def get_sigma(self, t):
        return self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t) ** self.power