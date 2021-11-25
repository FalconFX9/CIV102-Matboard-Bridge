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
        self.w = self.width
        self.h = self.height
        self.y = d_bottom
        self.x = x_pos

    def is_touching(self, rect):
        if self.x_pos <= rect.x_pos <= self.x_pos + self.width or \
                self.x_pos <= rect.x_pos + rect.width <= self.x_pos + self.width:
            if self.d_bottom + self.height == rect.d_bottom:

                return min(self.width, rect.width), self.d_bottom + self.height

            elif self.d_bottom == rect.d_bottom + rect.height:
                return min(self.width, rect.width), self.d_bottom

        return None, None

    def is_touching_horizontal(self, rect):
        if self.d_bottom <= rect.d_bottom <= self.d_bottom + self.height or \
                self.d_bottom <= rect.d_bottom + rect.height <= self.d_bottom + self.height:

            if self.x_pos + self.width == rect.x_pos:
                return min(self.height, rect.height), self.x_pos + self.width

            elif self.x_pos == rect.x_pos + rect.width:
                return min(self.height, rect.height), self.x_pos

        return None, None

    def get_properties(self):
        return self.w, self.h, self.y, self.x


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
    def calculate_centroid(section):
        """ Calculates and returns the centroid of a given cross-section"""
        nominator = 0
        area_total = 0
        for rect in section:
            h = abs(rect.h)
            area_section = abs(rect.w * h)
            nominator += area_section * (abs(rect.y) + (h / 2))
            area_total += area_section
        return nominator / area_total

    def get_i_section(self):
        """ Calculates and returns the second moment of area of a given cross-section"""
        I = 0
        for rect in self.sections:
            h = abs(rect.h)
            I += self.i_rect(rect.w, h) + (rect.w * h) * (abs(rect.y + (h / 2) - self.centroid) ** 2)
        return I

    def calculate_q_section(self):
        """ Calculates and returns the first moment of area of a given cross-section"""
        max_height = 0
        for rect in self.sections:
            if rect.h + rect.y > max_height:
                max_height = rect.h + rect.y

        precision = C.PRECISION
        for i in range(1, int(max_height * precision)):
            A = 0
            rects = []
            for rect in self.sections:
                rect_top = rect.h + rect.y
                rect_bottom = rect.y
                if rect_top <= i / precision:
                    rects.append(rect)
                elif rect_top > i / precision > rect_bottom:
                    # [width, height, distance from bottom of section to bottom of rectangle, x-position (left side))
                    sliced_rect = Rectangle(rect.w, (i / precision - rect_bottom), rect.y, rect.x)
                    rects.append(sliced_rect)

            centroid_Q_area = self.calculate_centroid(rects)

            d = abs(self.centroid - centroid_Q_area)
            for rect in rects:
                A += rect.w * rect.h

            self.Q.append(A * d)

    def get_Q(self):
        return self.Q

    def get_max_Q(self):
        maximum = max(self.Q)
        index = self.Q.index(maximum)
        return maximum, index

    def get_separated_deck(self):
        """ Returns Rect objects that represent the geometrically split up deck, and which case of plate they are
        Is a horrible mess.
        Needs to be optimized.
        No idea how general this is, but it works for the pi beam."""
        contact_areas = {}
        for n, rect in enumerate(self.sections[:-1]):
            for other_rect in self.sections[n + 1:]:
                contact_area, y = rect.is_touching(other_rect)
                if contact_area and y:
                    if rect not in contact_areas.keys():
                        contact_areas[rect] = []
                    contact_areas[rect].append([contact_area, y, other_rect])


        joint_rects = []
        plate_rects = []
        x_starts_ends = []
        case_n = []
        for rect in contact_areas.keys():
            print(rect.get_properties())
            temp_rect = Rectangle(rect.w, rect.h, rect.y, rect.x)
            for elements in contact_areas[rect]:
                joint_rects.append(Rectangle(elements[2].w, rect.h, rect.y, elements[2].x))
                x_starts_ends.append(elements[2].x)
                x_starts_ends.append(elements[2].x + elements[2].w)
            x_starts_ends.sort()
            for x_coord in x_starts_ends:
                n_width = x_coord - temp_rect.x
                n_rect = Rectangle(n_width, temp_rect.h, temp_rect.y, temp_rect.x)
                temp_rect.w -= n_width
                temp_rect.x = x_coord
                plate_rects.append(n_rect)

            c = 0
            delete_list = []
            for rect_p in plate_rects:
                for rect_joint in joint_rects:
                    if round(plate_rects[c].w, 2) == round(rect_joint.w, 2) and round(plate_rects[c].x, 2) == round(rect_joint.x, 2):
                        delete_list.append(rect_p)
                    elif round(plate_rects[c].w, 2) == 0:
                        delete_list.append(rect_p)
                c += 1
            for rect_d in delete_list:
                if rect_d in plate_rects:
                    plate_rects.remove(rect_d)

            for rect_p in plate_rects:
                n = 0
                for rect_joint in joint_rects:
                    j, k = rect_p.is_touching_horizontal(rect_joint)
                    print(j, k)
                    if j:
                        n += 1
                if n == 1:
                    case_n.append(2)
                elif n == 2:
                    case_n.append(1)

        return plate_rects, case_n








class BridgeSolver:

    def __init__(self, cross_sections, SFD, BMD):
        self.cross_sections = cross_sections
        self.Is = []
        self.centroids = []
        self.Qs = []
        self.QsAllY = []
        self.solve_section_properties()
        self.V_fail = []
        self.P_fail_C = []
        self.P_fail_T = []
        self.V_fail_MAT = []
        self.V_fail_glue = []
        self.SFD = SFD
        self.BMD = BMD

    def solve_section_properties(self):
        for cross_section in self.cross_sections:
            solver = CrossSectionSolver(cross_section)
            self.Is.append(solver.get_i_section())
            self.Qs.append(solver.get_max_Q())
            self.QsAllY.append(solver.get_Q())
            self.centroids.append(solver.centroid)

    def flex_failure(self):
        for x in range(C.BRIDGE_LENGTH):
            if self.BMD[x] > 0:
                self.P_fail_C.append(min(
                    ((C.SigC * self.Is[x]) / (self.cross_sections[x][0].y - self.centroids[x])) / (self.BMD[x] / C.P), C.MAX_FORCE))
                self.P_fail_T.append(min(C.MAX_FORCE, ((C.SigT * self.Is[x]) / (self.centroids[x])) / (self.BMD[x] / C.P)))
            elif self.BMD[x] < 0:
                self.P_fail_T.append(min(C.MAX_FORCE, ((C.SigC * self.Is[x]) / abs(
                    (self.cross_sections[x][1].y + self.cross_sections[x][1].h) - self.centroids[x])) / (
                                                 -self.BMD[x] / C.P)))
                self.P_fail_C.append(min(C.MAX_FORCE, ((C.SigT * self.Is[x]) / (self.centroids[x])) / (-self.BMD[x] / C.P)))

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
        # Sweep though height
        # Calculate overlapping rectangles
        # Note: Max tab width < A
        default_len = len(self.cross_sections[0])
        max_tab_width = 30
        tab_counter = []
        for x in range(C.BRIDGE_LENGTH):
            contact_areas = {}
            for i, rect in enumerate(self.cross_sections[x][:-1]):
                for other_rects in self.cross_sections[x][i + 1:]:
                    contact_area, y_glue = rect.is_touching(other_rects)
                    if contact_area and y_glue:
                        if y_glue in contact_areas.keys():
                            contact_areas[y_glue].append(contact_area)
                        else:
                            contact_areas[y_glue] = [contact_area]

            v_fail_glue_section = []
            min_sum = 100000000000000000000000000000000000000
            y_smallest = 0
            for y, contact_area in contact_areas.items():
                if sum(contact_area) < min_sum:
                    min_sum = sum(contact_area)
                    y_smallest = y
                v_fail = (C.TauG * self.Is[x] * sum(contact_area))/self.QsAllY[x][int(y*C.PRECISION) - 1]
                v_fail_glue_section.append(v_fail)

            try:
                self.V_fail_glue.append((min(v_fail_glue_section)/len(contact_areas[y_smallest])) / (abs(self.SFD[x]) / C.P))
            except ZeroDivisionError:
                pass
            """
            if len(section) > default_len:
                pass
                tab_counter.append()
            if tab_counter > max_tab_width:
                default_len = len(section)
                tab_counter = []
            else:
            """

    def plot(self):
        print(f"Min P Fail Compression (flexural): {min(self.P_fail_C)} (N)")
        print(f"Min P Fail Tension (flexural): {min(self.P_fail_T)} (N)")
        print(f"Min P Fail Matboard (shear): {min(self.V_fail_MAT)} (N)")
        print(f"Min P Fail Glue (shear): {min(self.V_fail_glue)} (N)")
        fig, axs = plt.subplots(1, 4)

        axs[0].plot(self.P_fail_C)
        axs[1].plot(self.P_fail_T)
        axs[2].plot(self.V_fail_MAT)
        axs[3].plot(self.V_fail_glue)
        axs[0].legend(["P Fail Compression"])
        axs[1].legend(["P Fail Tension"])
        axs[2].legend(["V Fail Matboard"])
        axs[3].legend(["V Fail Glue"])
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
        arch_rect_2 = [1.27, 90, 0, (100-1.27)]
        tab_1 = [10, 1.27, (90 - 1.27), 1.27]
        tab_2 = [10, 1.27, (90 - 1.27), (100-1.27-10)]
        cross_sections.append([Rectangle(deck[0], deck[1], deck[2], deck[3]),
                               Rectangle(arch_rect[0], arch_rect[1], arch_rect[2], arch_rect[3]),
                               Rectangle(arch_rect_2[0], arch_rect_2[1], arch_rect_2[2], arch_rect_2[3]),
                               Rectangle(tab_1[0], tab_1[1], tab_1[2], tab_1[3]),
                               Rectangle(tab_2[0], tab_2[1], tab_2[2], tab_2[3])])
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
    cs = CrossSectionSolver(cross_sections[0])
    cs.get_separated_plates()
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
    bridge_solver.glue_fail()
    bridge_solver.plot()
