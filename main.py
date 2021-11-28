"""
TODO: Implement plate buckling
"""

import matplotlib.pyplot as plt
import constants as C
import math


class Rectangle:

    def __init__(self, width, height, d_bottom, x_pos, ID=None):
        self.width = width
        self.height = height
        self.d_bottom = d_bottom
        self.x_pos = x_pos
        self.w = self.width
        self.h = self.height
        self.y = d_bottom
        self.x = x_pos
        self.ID = ID

    def is_touching(self, rect):
        if self.x_pos <= rect.x_pos <= self.x_pos + self.width or \
                self.x_pos <= rect.x_pos + rect.width <= self.x_pos + self.width:
            if rect.d_bottom - 0.1 <= self.d_bottom + self.height <= rect.d_bottom + 0.1:

                return min(self.width, rect.width), self.d_bottom + self.height

            elif self.d_bottom - 0.1 <= rect.d_bottom + rect.height <= self.d_bottom + 0.1:
                return min(self.width, rect.width), self.d_bottom

        return None, None

    def is_touching_horizontal(self, rect):
        if self.d_bottom - 0.1 <= rect.d_bottom <= self.d_bottom + self.height + 0.1 or \
                self.d_bottom - 0.1 <= rect.d_bottom + rect.height <= self.d_bottom + self.height + 0.1:

            if rect.x_pos - 0.1 <= self.x_pos + self.width <= rect.x_pos + 0.1:
                return min(self.height, rect.height), self.x_pos + self.width

            elif self.x_pos - 0.1 <= rect.x_pos + rect.width <= self.x_pos + 0.1:
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
      
        self.P1_train = [x+52 for x in range(291)]
        self.P2_train = [x+176 for x in self.P1_train]
        self.P3_train = [x+164 for x in self.P2_train]
        self.P4_train = [x+176 for x in self.P3_train]
        self.P5_train = [x+164 for x in self.P4_train]
        self.P6_train = [x+176 for x in self.P5_train]
        self.P_train = 400/6
        self.Fby_train = [(self.P1_train[x]*400/6 + self.P2_train[x]*400/6 + self.P3_train[x]*400/6 + self.P4_train[x]*400/6\
                           + self.P5_train[x]*400/6 + self.P6_train[x]*400/6)/self.B for x in range(291)]
        self.Fay_train = [400-x for x in self.Fby_train]
     
    def shear_force_train(self, x, i):
        """ Returns the shear force at x mm from support A in N, when the back of the train is at position i"""
        if x < self.B:
            if x < self.A:
                return 0
            elif self.A <= x < self.P1_train[i]:
                return self.Fay_train[i]
            elif self.P1_train[i] <= x < self.P2_train[i]:
                return self.Fay_train[i] - self.P_train
            elif self.P2_train[i] <= x < self.P3_train[i]:
                return self.Fay_train[i] - self.P_train*2
            elif self.P3_train[i] <= x < self.P4_train[i]:
                return self.Fay_train[i] - self.P_train*3
            elif self.P4_train[i] <= x < self.P5_train[i]:
                return self.Fay_train[i] - self.P_train*4
            elif self.P5_train[i] <= x < self.P6_train[i]:
                return self.Fay_train[i] - self.P_train*5
            elif self.P6_train[i] <= x:
                return self.Fay_train[i] - 400
        else:
            if self.P5_train[i] <= x < self.P6_train[i]:
                return 400 - self.P_train*5
            elif self.P6_train[i] <= x:
                return 0
        
    def moment_train(self, x, i):
        """ Returns the moment at x mm from support A in Nmm, when the back of the train is at position i"""
        if x < self.B:
            if x < self.A:
                return 0
            elif self.A <= x < self.P1_train[i]:
                return self.Fay_train[i] * x
            elif self.P1_train[i] <= x < self.P2_train[i]:
                return (self.Fay_train[i] * self.P1_train[i]) + (self.Fay_train[i] - self.P_train) * (x - self.P1_train[i])
            elif self.P2_train[i] <= x < self.P3_train[i]:
                return (self.Fay_train[i] * self.P1_train[i]) + (self.Fay_train[i] - self.P_train) * 176 + (self.Fay_train[i] - self.P_train*2) * (x - self.P2_train[i])
            elif self.P3_train[i] <= x < self.P4_train[i]:
                return ((self.Fay_train[i]*self.P1_train[i]) + (self.Fay_train[i]-self.P_train)*176 + (self.Fay_train[i]-self.P_train*2)*164\
                        + (self.Fay_train[i]-self.P_train*3)*(x-self.P3_train[i]))
            elif self.P4_train[i] <= x < self.P5_train[i]:
                return ((self.Fay_train[i]*self.P1_train[i]) + (self.Fay_train[i]-self.P_train)*176 + (self.Fay_train[i]-self.P_train*2)*164\
                        + (self.Fay_train[i]-self.P_train*3)*176 + (self.Fay_train[i]-self.P_train*4)*(x-self.P4_train[i]))
            elif self.P5_train[i] <= x < self.P6_train[i]:
                return ((self.Fay_train[i]*self.P1_train[i]) + (self.Fay_train[i]-self.P_train)*176 + (self.Fay_train[i]-self.P_train*2)*164\
                        + (self.Fay_train[i]-self.P_train*3)*176 + (self.Fay_train[i]-self.P_train*4)*164 + (self.Fay_train[i]-self.P_train*5)*(x-self.P5_train[i]))
            elif self.P6_train[i] <= x:
                return ((self.Fay_train[i]*self.P1_train[i]) + (self.Fay_train[i]-self.P_train)*176 + (self.Fay_train[i]-self.P_train*2)*164\
                        + (self.Fay_train[i]-self.P_train*3)*176 + (self.Fay_train[i]-self.P_train*4)*164 + (self.Fay_train[i]-self.P_train*5)*176))
        else:
            if self.P5_train[i] <= x < self.P6_train[i]:
                return ((self.Fay_train[i]*self.P1_train[i])+(self.Fay_train[i]-self.P_train)*176+(self.Fay_train[i]-self.P_train*2)*(self.B-self.P2_train[i])\
                        +(400-self.P_train*2)*(self.P3_train[i]-self.B)+(400-self.P_train*3)*164\
                        +(400-self.P_train*4)*176+(400-self.P_train*5)*(x-self.P5_train[i]))
            elif self.P6_train[i] <= x:
                return 0

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
        """ Returns the moment at x mm from support A in Nmm"""
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

    def plot_diagrams(self, i):
        """ Plots the shear and bending moment diagrams"""
        SFD = []
        BMD = []
        SFD_train = []
        BMD_train = []
        for x in range(1280):
            SFD.append(self.shear_force(x))
            BMD.append(self.moment(x))
            SFD_train.append(self.shear_force_train(x, i))
            BMD_train.append(self.moment_train(x, i))

        # using the variable axs for multiple Axes
        fig, axs = plt.subplots(1, 4)

        # using tuple unpacking for multiple Axes
        axs[0].plot(SFD)
        axs[1].plot(BMD)
        axs[1].invert_yaxis()
        axs[0].legend(["Shear Force (N)"])
        axs[1].legend(["Moment (Nmm)"])
        axs[2].plot(SFD_train)
        axs[3].plot(BMD_train)
        axs[3].invert_yaxis()
        axs[2].legend(["Train Shear Force (N)"])
        axs[3].legend(["Train Moment (Nmm)"])
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
        numerator = 0
        area_total = 0
        for rect in section:
            h = abs(rect.h)
            #print(rect.h, "Centroid")
            area_section = rect.w * h
            numerator += area_section * (abs(rect.y) + (h / 2))
            area_total += area_section

        return numerator / area_total

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
            #print(rect.h, 'Q')
        precision = C.PRECISION
        for i in range(1, int(max_height * precision)):
            A = 0
            rects = []
            for rect in self.sections:
                rect_top = rect.h + rect.y
                rect_bottom = rect.y
                if rect_top <= i / precision:
                    rects.append(Rectangle(rect.w, rect.h, rect.y, rect.x, rect.ID))
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
        contact_areas_y = {}
        for n, rect in enumerate(self.sections[:-1]):
            if rect.ID == C.DECK_ID:
                #print(rect.get_properties(), "RECT")
                for other_rect in list(self.sections[:n]+self.sections[n+1:]):
                    #print(other_rect.get_properties(), "OTHER_RECT")
                    contact_area, y = rect.is_touching(other_rect)

                    if contact_area and y:
                        if rect not in contact_areas.keys():
                            contact_areas[rect] = []
                        contact_areas[rect].append([contact_area, y, other_rect])

        joint_rects = []
        plate_rects_large = []
        x_starts_ends = {}
        case_n_large = []
        for rect in contact_areas.keys():
            temp_rect = Rectangle(rect.w, rect.h, rect.y, rect.x, rect.ID)
            for elements in contact_areas[rect]:
                joint_rects.append(Rectangle(elements[2].w, rect.h, rect.y, elements[2].x))
                if elements[1] not in x_starts_ends.keys():
                    x_starts_ends[elements[1]] = []
                x_starts_ends[elements[1]].append(elements[2].x)
                x_starts_ends[elements[1]].append(elements[2].x + elements[2].w)

            for y in x_starts_ends.keys():
                x_starts_ends[y] = list(set(x_starts_ends[y]))
                x_starts_ends[y].sort()

            for x_starts_end in x_starts_ends.values():
                plate_rects = []
                case_n = []
                for x_coord in x_starts_end:
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
                        if j:
                            n += 1
                    case_n.append(n)

                plate_rects_large.append(plate_rects)
                case_n_large.append(case_n)
        return plate_rects_large, case_n_large

    def get_sliced_webs(self):
        sliced_webs = []
        case_n = []
        for rect in self.sections:
            if rect.ID == 'WEB':
                sliced_web = Rectangle(rect.w, rect.d_bottom+rect.height-self.centroid, rect.d_bottom, rect.x_pos)
                sliced_webs.append(sliced_web)
                case_n.append(3)

        return sliced_webs, case_n


class BridgeSolver:

    def __init__(self, cross_sections, SFD, BMD):
        self.cross_sections = cross_sections
        self.Is = []
        self.centroids = []
        self.Qs = []
        self.QsAllY = []
        self.sliced_decks = []
        self.sliced_webs = []
        self.solve_section_properties()
        self.V_fail = []
        self.P_fail_C = []
        self.P_fail_T = []
        self.V_fail_MAT = []
        self.V_fail_glue = []
        self.V_fail_MAT_buckling = []
        self.P_fail_PLATE = []
        self.SFD = SFD
        self.BMD = BMD
        self.CMD = self.get_curvature_diagram()

    def get_curvature_diagram(self):
        curvs = []
        for x in range(C.BRIDGE_LENGTH):
            curvs.append(self.BMD[x] / (self.Is[x] * C.E))

        # print(curvs)
        return curvs

    def solve_section_properties(self):
        for cross_section in self.cross_sections:
            solver = CrossSectionSolver(cross_section)
            self.Is.append(solver.get_i_section())
            self.Qs.append(solver.get_max_Q())
            self.QsAllY.append(solver.get_Q())
            self.centroids.append(solver.centroid)
            self.sliced_decks.append(solver.get_separated_deck())
            self.sliced_webs.append(solver.get_sliced_webs())

    def flex_failure(self):
        for x in range(C.BRIDGE_LENGTH):
            if self.BMD[x] > 0:
                max_height = 0
                for rect in self.cross_sections[x]:
                    if rect.h + rect.y > max_height:
                        max_height = rect.h + rect.y
                self.P_fail_C.append(min(
                    ((C.SigC * self.Is[x]) / (max_height - self.centroids[x])) / abs(self.BMD[x]/C.P), C.MAX_FORCE))
                self.P_fail_T.append(min(C.MAX_FORCE, ((C.SigT * self.Is[x]) / abs(self.centroids[x])) / (self.BMD[x]/C.P)))
            elif self.BMD[x] < 0:
                max_height = 0
                for rect in self.cross_sections[x]:
                    if rect.h + rect.y > max_height:
                        max_height = rect.h + rect.y
                self.P_fail_C.append(min(C.MAX_FORCE, ((C.SigC * self.Is[x]) / abs(
                    self.centroids[x])) / abs(
                                                 self.BMD[x]/C.P)))
                self.P_fail_T.append(min(C.MAX_FORCE, ((C.SigT * self.Is[x]) / abs(max_height - self.centroids[x])) /abs(self.BMD[x]/C.P)))

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
                for other_rects in list(self.cross_sections[x][:i]+self.cross_sections[x][i+1:]):
                    if rect.ID == C.DECK_ID:
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
                min_Q = self.QsAllY[x][int(y*C.PRECISION)-1]
                v_fail = (C.TauG * self.Is[x] * sum(contact_area))/min_Q
                if y != C.NO_GLUE:
                    v_fail_glue_section.append(v_fail)

            try:
                self.V_fail_glue.append(min(v_fail_glue_section) / (abs(self.SFD[x]) / C.P))
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

    def plate_buckling(self):
        for x in range(C.BRIDGE_LENGTH):
            sigma_crit = 1000000000000000000
            sigma_crits = []
            for single_layer in self.sliced_decks[x]:
                for n in range(len(single_layer)):
                    sliced_deck = single_layer[n]
                    for i in range(len(sliced_deck)):
                        rect = self.sliced_decks[x][0][n][i]
                        case_n = self.sliced_decks[x][1][n][i]
                        if case_n == 1:
                            # print(f"Case 1: {rect.get_properties()}")
                            up_or_down = rect.y - self.centroids[x]
                            sigma_crits.append([self.case_1(rect.h, rect.w), up_or_down])

                            # print(f"Case 1 Sigma: {self.case_1(rect.h, rect.w)}")
                        elif case_n == 2:
                            # print(f"Case 2: {rect.get_properties()}")
                            up_or_down = rect.y - self.centroids[x]
                            sigma_crits.append([self.case_2(rect.h, rect.w), up_or_down])

                            # print(f"Case 2 Sigma: {self.case_2(rect.h, rect.w)}")

            for i in range(len(self.sliced_webs[x][0])):
                rect = self.sliced_webs[x][0][i]
                case_n = self.sliced_webs[x][1][i]
                if case_n == 3:
                    # print(f"Case 3: {rect.get_properties()}")
                    sigma_crit = min(self.case_3(rect.w, rect.h), sigma_crit)
                    # print(f"Case 3 Sigma: {self.case_3(rect.w, rect.h)}")
            if self.BMD[x] > 0:
                for sg in sigma_crits:
                    if sg[1] > 0:
                        sigma_crit = min(sg[0], sigma_crit)
            elif self.BMD[x] < 0:
                for sg in sigma_crits:
                    if sg[1] < 0:
                        sigma_crit = min(sg[0], sigma_crit)

            if self.BMD[x] > 0:
                max_height = 0
                for rect in self.cross_sections[x]:
                    if rect.h + rect.y > max_height:
                        max_height = rect.h + rect.y
                P_fail = min((
                    ((sigma_crit * self.Is[x]) / (max_height - self.centroids[x])) / abs(self.BMD[x] / C.P),
                    C.MAX_FORCE))
            elif self.BMD[x] < 0:
                max_height = 0
                for rect in self.cross_sections[x]:
                    if rect.h + rect.y > max_height:
                        max_height = rect.h + rect.y
                P_fail = min(C.MAX_FORCE, ((sigma_crit * self.Is[x]) / abs(
                    self.centroids[x])) / abs(
                    self.BMD[x] / C.P))
            else:
                P_fail = C.MAX_FORCE
            self.P_fail_PLATE.append(P_fail)

    @staticmethod
    def case_1(t, b):
        return ((4 * (math.pi ** 2) * C.E) / (12 * (1 - C.mu ** 2))) * ((t / b) ** 2)

    @staticmethod
    def case_2(t, b):
        return ((0.425 * (math.pi ** 2) * C.E) / (12 * (1 - C.mu ** 2))) * ((t / b) ** 2)

    @staticmethod
    def case_3(t, b):
        return ((6 * (math.pi ** 2) * C.E) / (12 * (1 - C.mu ** 2))) * ((t / b) ** 2)

    @staticmethod
    def get_shear_buckling(t, h, a):
        return ((5*(math.pi**2)*C.E)/(12*(1-C.mu**2))) * ((t / h) ** 2 + (t / a) ** 2)

    def shear_buckling(self):
        for x in range(C.BRIDGE_LENGTH):
            h_web = 0
            for rect in self.cross_sections[x]:
                h_web = max(rect.h, h_web)
            a = 550  # temporary
            tau = self.get_shear_buckling(C.DIAPHRAGM_THICKNESS, h_web, a)
            q, y = self.Qs[x]
            t = C.MATBOARD_THICKNESS  # CALCULATE IN CROSS-SECTION SOLVER
            try:
                self.V_fail_MAT_buckling.append(((tau * self.Is[x] * t * 2) / q) / (abs(self.SFD[x]) / C.P))
            except ZeroDivisionError:
                pass

    def midspan_deflection(self):
        max_curvature = max(self.CMD)
        min_curvature = min(self.CMD)

        # print(C.P, max_curvature, min_curvature, )

        span_L = 1060
        load_L = 550

        # BELOW: Area of triangle from support A to midspan times D (distance from midspan to centroid)
        triangle_to_mid = (0.5 * self.CMD[span_L // 2] * (span_L // 2)) * ((span_L // 2) / 3)

        # BELOW: Area of triangle from support to change in sign of slope of curvature diagram
        triangle_1 = (0.5 * max_curvature * load_L) * ((span_L - load_L) + load_L / 3)

        # Below: Area of the min triangle moved down by the max and added to a rectangle to normalize it back to center
        triangle_2 = (0.5 * (min_curvature - max_curvature) * (span_L - load_L)) * ((span_L - load_L) / 3) + (
                    (span_L - load_L) * (max_curvature)) * ((span_L - load_L) / 2)

        triangle_to_support = triangle_1 + triangle_2

        # Always 1/2 because considering midspan, doesn't depend on span size
        D_mid = ((1 / 2 * triangle_to_support - triangle_to_mid))
        print(D_mid)
            
    def FOS(self):
        '''solves for FOS of Design 0 under Load case 1'''
        pass
      

    def plot(self):
        min_P_comp = min(self.P_fail_C)
        min_P_tens = min(self.P_fail_T)
        min_P_sMAT = min(self.V_fail_MAT)
        min_P_sGLUE = min(self.V_fail_glue)
        min_P_sbMATH = min(self.V_fail_MAT_buckling)
        min_P_PLATE = min(self.P_fail_PLATE)
        print(f"Min P Fail Compression (flexural): {min_P_comp} N, at x={self.P_fail_C.index(min_P_comp)}")
        print(f"Min P Fail Tension (flexural): {min_P_tens} N, at x={self.P_fail_T.index(min_P_tens)}")
        print(f"Min P Fail Matboard (shear): {min_P_sMAT} N, at x={self.V_fail_MAT.index(min_P_sMAT)}")
        print(f"Min P Fail Glue (shear): {min_P_sGLUE} N, at x={self.V_fail_glue.index(min_P_sGLUE)}")
        print(f"Min P Fail Shear Buckling: {min_P_sbMATH} N, at x={self.V_fail_MAT_buckling.index(min_P_sbMATH)}")
        print(f"Min P Fail Plate Buckling: {min_P_PLATE} N, at x={self.P_fail_PLATE.index(min_P_PLATE)}")
        min_any = min(min_P_comp, min_P_tens, min_P_sMAT, min_P_sGLUE, min_P_sbMATH, min_P_PLATE)
        print(f"The bridge will fail at {int(round(min_any, 0))} N")
        fig, axs = plt.subplots(1, 6)

        axs[0].plot(self.P_fail_C)
        axs[1].plot(self.P_fail_T)
        axs[2].plot(self.V_fail_MAT)
        axs[3].plot(self.V_fail_glue)
        axs[4].plot(self.V_fail_MAT_buckling)
        axs[5].plot(self.P_fail_PLATE)
        axs[0].legend(["P Fail Compression"])
        axs[1].legend(["P Fail Tension"])
        axs[2].legend(["V Fail Matboard"])
        axs[3].legend(["V Fail Glue"])
        axs[4].legend(["V Fail MAT shear buckling"])
        axs[5].legend(["P Fail Plate"])
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
        y_upper = arch.over_arch(x)
        deck = [100, 1.27, 90, 0]
        arch_rect = [1.27, 90, 0, 10]
        arch_rect_2 = [1.27, 90, 0, (90-1.27)]
        tab_1 = [10, 1.27, (90 - 1.27), 10+1.27]
        tab_2 = [10, 1.27, (90 - 1.27), (90-1.27-10)]
        """
        if y_upper > 0:
            cross_sections.append([Rectangle(deck[0], deck[1], deck[2], deck[3], C.DECK_ID),
                                   Rectangle(arch_rect[0], arch_rect[1], arch_rect[2], arch_rect[3], C.WEB_ID),
                                   Rectangle(arch_rect_2[0], arch_rect_2[1], arch_rect_2[2], arch_rect_2[3], C.WEB_ID),
                                   Rectangle(tab_1[0], tab_1[1], tab_1[2], tab_1[3], C.GLUE_TAB_ID),
                                   Rectangle(tab_2[0], tab_2[1], tab_2[2], tab_2[3], C.GLUE_TAB_ID),
                                   Rectangle(arch_rect[0], y_upper, 91.27, 10),
                                   Rectangle(arch_rect[0], y_upper, 91.27, 90-1.27),
                                   Rectangle(tab_1[0], tab_1[1], 91.27, tab_1[3], C.GLUE_TAB_ID),
                                   Rectangle(tab_2[0], tab_2[1], 91.27, tab_2[3], C.GLUE_TAB_ID),
                                   ])
        else:
        cross_sections.append([Rectangle(deck[0], deck[1], deck[2], deck[3], C.DECK_ID),
                               Rectangle(arch_rect[0], arch_rect[1], arch_rect[2], arch_rect[3], C.WEB_ID),
                               Rectangle(arch_rect_2[0], arch_rect_2[1], arch_rect_2[2], arch_rect_2[3], C.WEB_ID),
                               Rectangle(tab_1[0], tab_1[1], tab_1[2], tab_1[3], C.GLUE_TAB_ID),
                               Rectangle(tab_2[0], tab_2[1], tab_2[2], tab_2[3], C.GLUE_TAB_ID)
                               ])
        """
        t = 1.27
        deck = [100, t, 75-t, 0]  # top is 75+t
        tab_1 = [10, t, (75 - 2*t), 10 + t]
        tab_2 = [10, t, (75 - 2* t), (90 - t - 10)]

        cross_sections.append([Rectangle(deck[0], 1.27, deck[2], deck[3], C.DECK_ID),  # Deck
                               Rectangle(t, (deck[2]-t), t, 10, C.WEB_ID),  # Web1
                               Rectangle(t, (deck[2]-t), t, (90 - t), C.WEB_ID),  # Web2
                               Rectangle(80, t, 0, 10+t, C.DECK_ID),  # bottom
                               Rectangle(tab_1[0], tab_1[1], tab_1[2], tab_1[3], C.GLUE_TAB_ID),  # Glue_tab1
                               Rectangle(tab_2[0], tab_2[1], tab_2[2], tab_2[3], C.GLUE_TAB_ID)])  # Glue tab 2
    """
        y_under = arch.under_arch(x)
        
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
    diagrams = Diagrams(C.P)
    diagrams.plot_diagrams(0)
    cross_sections = generate_cross_sections(Arch())
    #cs = CrossSectionSolver(cross_sections[0])
    #cs.get_separated_plates()
    SFD = []
    BMD = []
    import c_s_visualizer
    #cs = c_s_visualizer.DrawCrossSection(cross_sections, None)
    #cs.draw_animation()
    #c_s_visualizer.DrawElevation(cross_sections).draw(549, None)
    for x in range(1280):
        SFD.append(diagrams.shear_force(x))
        BMD.append(diagrams.moment(x))
    bridge_solver = BridgeSolver(cross_sections, SFD, BMD)
    bridge_solver.flex_failure()
    bridge_solver.shear_failure()
    bridge_solver.glue_fail()
    bridge_solver.shear_buckling()
    bridge_solver.plate_buckling()
    bridge_solver.midspan_deflection()
    bridge_solver.plot()
