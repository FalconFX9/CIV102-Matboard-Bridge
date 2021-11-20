import matplotlib.pyplot as plt


class Diagrams:
    """ Generates the composite functions representing both moment diagrams based on a given P
    """
    def __init__(self, P):
        self.A = 0
        self.P1 = 550
        self.B = self.P1 + 510
        self.P2 = self.B + 190
        self.P = P
        self.Fby = (self.P2 * P + (self.P1 * P)) / self.B
        self.Fay = P * 2 - self.Fby

    def shear_force(self, x):
        """ Returns the shear force at x mm from support A in N"""
        if x < self.A:
            return 0
        elif self.A <= x < self.P1:
            return self.Fay
        elif self.P1 <= x < self.B:
            return self.Fay - self.P
        elif self.B <= x < self.P2:
            return self.Fay - self.P + self.Fby
        elif self.P2 <= x:
            return 0

    def moment(self, x):
        """ Returns the moment at x mm from support A in Nm"""
        if x < self.A:
            return 0
        elif self.A <= x < self.P1:
            return self.Fay * x
        elif self.P1 <= x < self.B:
            return (self.Fay * self.P1) + (self.Fay - self.P) * (x-self.P1)
        elif self.B <= x < self.P2:
            return (self.Fay * self.P1) + (self.Fay - self.P) * (self.B-self.P1) + (self.Fay - self.P + self.Fby) * (x-self.B)
        elif self.P2 <= x:
            return 0

    def plot_diagrams(self):
        SFD = []
        BMD = []
        for x in range(1280):
            SFD.append(self.shear_force(x))
            BMD.append(self.moment(x))

        # using the variable axs for multiple Axes
        fig, axs = plt.subplots(1, 2)

        # using tuple unpacking for multiple Axes
        axs[0].plot(SFD)
        axs[1].plot(BMD)
        axs[1].invert_yaxis()
        axs[0].legend(["Shear Force (N)"])
        axs[1].legend(["Moment (Nmm)"])
        plt.show()


class CrossSectionSolver:
    """ Takes in a list of the rectangles of a split-up cross section in the form (B, H, Y_section)
    All 3 in mm
    Y_section represents the distance between lower edge of the rectangle and the bottom of the cross-section
    """
    def __init__(self, sections):
        self.sections = sections
        self.centroid = 0
        self.calculate_centroid()

    @staticmethod
    def i_rect(b, h):
        """
        :param b: Width of the rectangle
        :param h: Height of the rectangle
        :return: Second moment of area of the rectangle
        """
        return (b*h**3)/12

    def calculate_centroid(self):
        """ Calculates and returns the centroid of a given cross-section"""
        nominator = 0
        area_total = 0
        for section in self.sections:
            b, h, y_section = section
            area_section = b * h
            nominator += area_section * (y_section + (h/2))
            area_total += area_section
        return nominator/area_total

    def get_i_section(self):
        """ Calculates and returns the second moment of area of a given cross-section"""
        I = 0
        for section in self.sections:
            b, h, y_section = section
            I += self.i_rect(b, h) + (b*h) * (abs(y_section + (h/2)-self.centroid) ** 2)
        return I


if __name__ == "__main__":
    P = 500
    diagrams = Diagrams(P)
    diagrams.plot_diagrams()
