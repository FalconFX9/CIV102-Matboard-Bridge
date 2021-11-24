"""
TODO: Add variable cross-section functionality
"""


import matplotlib.pyplot as plt
import constants as C


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
        """ Plots the shear and bending moment diagrams"""
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

        self.centroid = self.calculate_centroid()

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
            b, h, y_section, x = section
            h = abs(h)
            area_section = abs(b * h)
            nominator += area_section * (abs(y_section) + (h/2))
            area_total += area_section
        return nominator/area_total

    def get_i_section(self):
        """ Calculates and returns the second moment of area of a given cross-section"""
        I = 0
        for rect in self.sections:
            b, h, y_section, x = rect
            h = abs(h)
            I += self.i_rect(b, h) + (b*h) * (abs(y_section + (h/2)-self.centroid) ** 2)
        return I

    def get_q_section(self):
        """ Calculates and returns the first moment of area of a given cross-section"""
        Q = 0
        for rect in self.sections:
            b, h, y_section, x = rect
            h = abs(h)
            Q += (b * h) * (abs(y_section + (h / 2) - self.centroid) ** 2)

        return Q


class BridgeSolver:

    def __init__(self, cross_sections, SFD, BMD):
        self.cross_sections = cross_sections
        self.Is = []
        self.centroids = []
        self.Qs = []
        self.solve_section_properties()
        
        # cause of failure: shear failure (matboard or glue)
        self.P_V_fail_G = []
        self.P_V_fail_B = []
        
        # cause of failure: plate buckling
        self.P_fail_buck = []
        
        # cause of failure: shear buckling
        self.P_V_fail_buck = []
        
        # cause of failure: flexual failures (compression/tension)
        self.P_fail_C = []
        self.P_fail_T = []
        
        self.SFD = SFD
        self.BMD = BMD
        

    def solve_section_properties(self):
        for cross_section in self.cross_sections:
            solver = CrossSectionSolver(cross_section)
            self.Is.append(solver.get_i_section())
            # self.Qs.append(solver.get_q_section())
            self.centroids.append(solver.centroid)

    def flex_failure(self):
        for x in range(C.BRIDGE_LENGTH):
            if self.BMD[x] > 0:
                self.P_fail_C.append(((C.SigC * self.Is[x])/(self.cross_sections[x][0][2]-self.centroids[x]))/(self.BMD[x]/500))
                self.P_fail_T.append(((C.SigT * self.Is[x]) / (self.centroids[x])) / (self.BMD[x]/500))
            elif self.BMD[x] < 0:
                self.P_fail_T.append(((C.SigC * self.Is[x])/abs((self.cross_sections[x][1][2] + self.cross_sections[x][1][1])-self.centroids[x]))/(-self.BMD[x]/500))
                self.P_fail_C.append(((C.SigT * self.Is[x]) / (self.centroids[x])) / (-self.BMD[x]/500))

    def shear_failure(self):
        # for a cross section at location x:
        if self.BMD[x] > 0:
            pass
        elif self.BMD[x] < 0:
            pass
        pass
    
    def plate_buckling(self):
        pass
    
    def shear_buckling(self):
        # for distance x between 2 diaphragms
        
        pass
                
    def plot(self):
        fig, axs = plt.subplots(1, 2)

        axs[0].plot(self.P_fail_C)
        axs[1].plot(self.P_fail_T)
        axs[0].legend(["P Fail Compression"])
        axs[1].legend(["P Fail Tension"])
        plt.show()


class Arch:

    @staticmethod
    def under_arch(x):
        y = 0.0016 * x * (x-788) - 50
        if y < 0:
            return y
        else:
            return 0

    @staticmethod
    def over_arch(x):
        y = -0.0017 * (x - 788) * (x - 1280)
        if y > 0:
            return y
        else:
            return 0


def generate_cross_sections(arch):
    bridge_length = 1280
    cross_sections = []
    deck = [100, 1.27, 0]
    for x in range(bridge_length):
        y_under = arch.under_arch(x)
        y_upper = arch.over_arch(x)
        if y_under < 0:
            arch_rect = [1.27, y_under-y_upper, 0, 0]
            cross_sections.append([[deck[0], deck[1], abs(y_under), 0], arch_rect, [arch_rect[0], arch_rect[1], arch_rect[2], 98], [deck[0], deck[1], 0, 0]])
        else:
            arch_rect = [1.27, y_upper, 0, 0]
            cross_sections.append(
                [[deck[0], deck[1], 0, 0], arch_rect, [arch_rect[0], arch_rect[1], arch_rect[2], 98]])

    return cross_sections


if __name__ == "__main__":
    P = 500
    diagrams = Diagrams(P)
    diagrams.plot_diagrams()
    cross_sections = generate_cross_sections(Arch())
    SFD = []
    BMD = []
    import c_s_visualizer
    cs = c_s_visualizer.DrawCrossSection(cross_sections, None)
    #cs.draw_animation()
    for x in range(1280):
        SFD.append(diagrams.shear_force(x))
        BMD.append(diagrams.moment(x))
    bridge_solver = BridgeSolver(cross_sections, SFD, BMD)
    bridge_solver.flex_failure()
    bridge_solver.plot()
