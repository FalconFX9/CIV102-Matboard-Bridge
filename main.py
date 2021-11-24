"""
TODO: Implement glue shear failure
      Implement plate buckling
      Implement shear buckling
"""

import matplotlib.pyplot as plt
import constants as C


class Rectangle:

    def __init__(self, width, height, d_bottom, x_pos):
        self.width = width
        self.height = height
        self.d_bottom = d_bottom
        self.x_pos = x_pos

    def is_touching(self, rect):
        if self.d_bottom < rect.d_bottom < self.d_bottom + self.height or \
                self.d_bottom < rect.d_bottom + rect.height < self.d_bottom + self.height:

            if self.x_pos + self.width == rect.x_pos:
                return min(self.height, rect.height)

        if self.x_pos < rect.x_pos < self.x_pos + self.width or \
                self.x_pos < rect.x_pos + rect.width < self.x_pos + self.width:

            if self.d_bottom + self.height == rect.d_bottom:
                return self.width + rect.width


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
            return (self.Fay * self.P1) + (self.Fay - self.P) * (x - self.P1)
        elif self.B <= x < self.P2:
            return (self.Fay * self.P1) + (self.Fay - self.P) * (self.B - self.P1) + (self.Fay - self.P + self.Fby) * (
                        x - self.B)
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

        self.centroid = self.calculate_centroid(self.sections)
        self.Q = []
        self.calculate_q_section()

    @staticmethod
    def i_rect(b, h):
        """
        :param b: Width of the rectangle
        :param h: Height of the rectangle
        :return: Second moment of area of the rectangle
        """
        return (b * h ** 3) / 12

    @staticmethod
    def calculate_centroid(sections):
        """ Calculates and returns the centroid of a given cross-section"""
        nominator = 0
        area_total = 0
        for section in sections:
            b, h, y_section, x = section
            h = abs(h)
            area_section = abs(b * h)
            nominator += area_section * (abs(y_section) + (h / 2))
            area_total += area_section
        return nominator / area_total

    def get_i_section(self):
        """ Calculates and returns the second moment of area of a given cross-section"""
        I = 0
        for rect in self.sections:
            b, h, y_section, x = rect
            h = abs(h)
            I += self.i_rect(b, h) + (b * h) * (abs(y_section + (h / 2) - self.centroid) ** 2)
        return I

    def calculate_q_section(self):
        """ Calculates and returns the first moment of area of a given cross-section"""
        max_height = 0
        for rect in self.sections:
            if rect[1] + rect[2] > max_height:
                max_height = rect[1] + rect[2]

        precision = 100
        for i in range(1, int(max_height * precision)):
            A = 0
            rects = []
            for rect in self.sections:
                rect_top = rect[1] + rect[2]
                rect_bottom = rect[2]
                if rect_top <= i / precision:
                    rects.append(rect)
                elif rect_top > i / precision > rect_bottom:
                    # [width, height, distance from bottom of section to bottom of rectangle, x-position (left side))
                    sliced_rect = [rect[0], (i / precision - rect_bottom), rect[2], rect[3]]
                    rects.append(sliced_rect)

            centroid_Q_area = self.calculate_centroid(rects)

            d = abs(self.centroid - centroid_Q_area)
            for rect in rects:
                A += rect[0] * rect[1]

            self.Q.append(A * d)

    def get_Q(self):
        return self.Q

    def get_max_Q(self):
        maximum = max(self.Q)
        index = self.Q.index(maximum)
        return maximum, index


class BridgeSolver:

    def __init__(self, cross_sections, SFD, BMD):
        self.cross_sections = cross_sections
        self.Is = []
        self.centroids = []
        self.Qs = []
        self.solve_section_properties()
        self.V_fail = []
        self.P_fail_C = []
        self.P_fail_T = []
        self.V_fail_MAT = []
        self.SFD = SFD
        self.BMD = BMD

    def solve_section_properties(self):
        for cross_section in self.cross_sections:
            solver = CrossSectionSolver(cross_section)
            self.Is.append(solver.get_i_section())
            self.Qs.append(solver.get_max_Q())
            self.centroids.append(solver.centroid)

    def flex_failure(self):
        for x in range(C.BRIDGE_LENGTH):
            if self.BMD[x] > 0:
                self.P_fail_C.append(
                    ((C.SigC * self.Is[x]) / (self.cross_sections[x][0][2] - self.centroids[x])) / (self.BMD[x] / C.P))
                self.P_fail_T.append(((C.SigT * self.Is[x]) / (self.centroids[x])) / (self.BMD[x] / C.P))
            elif self.BMD[x] < 0:
                self.P_fail_T.append(((C.SigC * self.Is[x]) / abs(
                    (self.cross_sections[x][1][2] + self.cross_sections[x][1][1]) - self.centroids[x])) / (
                                                 -self.BMD[x] / C.P))
                self.P_fail_C.append(((C.SigT * self.Is[x]) / (self.centroids[x])) / (-self.BMD[x] / C.P))

    def shear_failure(self):
        # Vfail = (Tfail * I * B) / Q
        for x in range(C.BRIDGE_LENGTH):
            q, y = self.Qs[x]
            t = 2 * 1.27  # CALCULATE IN CROSS-SECTION SOLVER
            try:
                self.V_fail_MAT.append(((C.TauM * self.Is[x] * t) / q) / (abs(self.SFD[x]) / C.P))
            except ZeroDivisionError:
                pass

    def glue_fail(self):
        pass
        # Sweep though height
        # Calculate overlapping rectangles

    def plot(self):
        fig, axs = plt.subplots(1, 3)

        axs[0].plot(self.P_fail_C)
        axs[1].plot(self.P_fail_T)
        axs[2].plot(self.V_fail_MAT)
        axs[0].legend(["P Fail Compression"])
        axs[1].legend(["P Fail Tension"])
        axs[2].legend(["V Fail Matboard"])
        plt.show()


class Arch:

    @staticmethod
    def under_arch(x):
        y = 0.0016 * x * (x - 788) - 50
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
        deck = [100, 1.27, 90, 0]
        arch_rect = [1.27, 90, 0, 0]
        arch_rect_2 = [1.27, 90, 0, 98]
        cross_sections.append([deck, arch_rect, arch_rect_2])
    """
        y_under = arch.under_arch(x)
        y_upper = arch.over_arch(x)
        if y_under < 0:
            arch_rect = [1.27, y_under-y_upper, 0, 0]
            cross_sections.append([[deck[0], deck[1], abs(y_under), 0], arch_rect, [arch_rect[0], arch_rect[1], arch_rect[2], 98], [deck[0], deck[1], 0, 0]])
        else:
            arch_rect = [1.27, y_upper, 0, 0]
            cross_sections.append(
                [[deck[0], deck[1], 0, 0], arch_rect, [arch_rect[0], arch_rect[1], arch_rect[2], 98]])
    """

    return cross_sections


if __name__ == "__main__":
    """
    deck = [100, 1.27, 0]
    arch_rect = [1.27, 90, 0, 0]
    section = [[deck[0], deck[1], 90, 0], arch_rect, [arch_rect[0], arch_rect[1], arch_rect[2], 98]]
    cs_solver = CrossSectionSolver(section)

    cs_solver.calculate_q_section()

    print(cs_solver.get_max_Q())
    print(cs_solver.centroid)
    """

    diagrams = Diagrams(C.P)
    diagrams.plot_diagrams()
    cross_sections = generate_cross_sections(Arch())
    SFD = []
    BMD = []
    # import c_s_visualizer
    # cs = c_s_visualizer.DrawCrossSection(cross_sections, None)
    # cs.draw_animation()
    for x in range(1280):
        SFD.append(diagrams.shear_force(x))
        BMD.append(diagrams.moment(x))
    bridge_solver = BridgeSolver(cross_sections, SFD, BMD)
    bridge_solver.flex_failure()
    bridge_solver.shear_failure()
    bridge_solver.plot()
