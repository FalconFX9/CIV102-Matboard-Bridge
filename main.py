"""
TODO: Implement plate buckling
"""

import matplotlib.pyplot as plt
import constants as C
import math

import sys

sys.setrecursionlimit(1300)


class Rectangle:

    def __init__(self, width, height, d_bottom, x_pos, ID=None):
        # All dimensions in [mm]
        self.width = width  # Width of rectangle
        self.height = height  # Height of rectangle
        self.d_bottom = d_bottom  # Distance from bottom of bounding box to bottom of rectangle
        self.x_pos = x_pos  # Distance from left of bounding box to left side of rectangle
        self.w = self.width  # Copies of above for quick access
        self.h = self.height
        self.y = d_bottom
        self.x = x_pos
        self.ID = ID  # Identification of the rectangle. Ex deck, web, etc. Check constants.py for values

    def is_touching(self, rect):
        """
        @param rect: Pass in a rectangle that is possibly congruent with self rectangle
        @return: If they are overlapping, it returns (overlap, height). Else, None, None
        """

        # Overlap on the x plane
        if self.x_pos <= rect.x_pos <= self.x_pos + self.width or \
                self.x_pos <= rect.x_pos + rect.width <= self.x_pos + self.width:
            # At the right height, other rect below self. 0.1 to account for floating point errors
            if rect.d_bottom - 0.1 <= self.d_bottom + self.height <= rect.d_bottom + 0.1:
                return min(self.width, rect.width), self.d_bottom + self.height

            # At the right height with other rect on bottom
            elif self.d_bottom - 0.1 <= rect.d_bottom + rect.height <= self.d_bottom + 0.1:
                return min(self.width, rect.width), self.d_bottom

        # Rectangles not touching
        return None, None

    def is_touching_horizontal(self, rect):
        """
        @param rect: Pass in a rectangle that is possibly congruent with self rectangle
        @return: If there is a vertical overlap (the touching side is vertical),
                 it returns (overlap, horizontal position). Else, None, None
        """

        # Are they overlapping on the vertical axis (y's overlap)
        if self.d_bottom - 0.1 <= rect.d_bottom <= self.d_bottom + self.height + 0.1 or \
                self.d_bottom - 0.1 <= rect.d_bottom + rect.height <= self.d_bottom + self.height + 0.1:

            # Are they beside each other (x's overlap). Self on left
            if rect.x_pos - 0.1 <= self.x_pos + self.width <= rect.x_pos + 0.1:
                return min(self.height, rect.height), self.x_pos + self.width

            # Are they beside each other (x's overlap). Self on right
            elif self.x_pos - 0.1 <= rect.x_pos + rect.width <= self.x_pos + 0.1:
                return min(self.height, rect.height), self.x_pos

        return None, None

    def get_properties(self):
        """
        @return: Properties of the rectangle in the order below
        """
        return self.w, self.h, self.y, self.x


class Diagrams_case1:
    """ Generates the functions representing moment and shear for the beeg train boi
    need to loop through all train x positions and take the maximum and minimum values for analysis of failure loads
    Method: Loop through each x and for each x put the train at every position possible. Take the max values
    and continue to the next x in the diagrams
    """

    def __init__(self):
        self.A = 0  # Distance from left to support 1
        self.B = 1060  # Distance from left to support 2 [mm]
        self.P = 400 / 6  # Load of the train [N] per pair of wheels
        self.w_s = 176  # Distance between train wheels [mm]
        self.c_s = 164  # Distance between train cars [mm]
        self.train_length = self.w_s * 3 + self.c_s * 2  # Total length of the train [mm]
        self.forceA = 0  # Reaction forces [N]
        self.forceB = 0
        self.BMD_steven_is_the_best_ta = {}  # Lookup table for previously calculated moments (since moment is
                                             # recursive, this dramatically improves runtime)
                                             # Aptly named because Steven has the beegest brain and can store all our values for us

    def reaction_forces(self, train_x):
        """
        Calculates the reaction forces given train position from the left
        @param train_x: position of the leftmost wheel of the train from the left side of the bridge
        """

        if train_x + self.train_length < 1280:  # If the train is completely on the bridge, all wheels cause
                                                # moment around point A and must be used to calculate B
            self.forceB = (self.P * train_x + self.P * (train_x + self.w_s) + self.P * (train_x + self.w_s + self.c_s) +
                           self.P * (train_x + 2 * self.w_s + self.c_s) + self.P * (
                                       train_x + 2 * self.w_s + 2 * self.c_s) +
                           self.P * (train_x + 3 * self.w_s + 2 * self.c_s)) / self.B

            self.forceA = self.P * 6 - self.forceB  # Force A is the weight of the train minus the force of B

        # Each if statement after this checks if one or more wheel is off the bridge. The only thing that changes each
        # time is that one wheel is removed from the moment consideration and the weight of the train is reduced by 1P
        elif train_x + self.train_length < 1280 + self.w_s:
            self.forceB = (self.P * train_x + self.P * (train_x + self.w_s) + self.P * (train_x + self.w_s + self.c_s) +
                           self.P * (train_x + 2 * self.w_s + self.c_s) + self.P * (
                                       train_x + 2 * self.w_s + 2 * self.c_s)
                           ) / self.B
            self.forceA = self.P * 5 - self.forceB
        elif train_x + self.train_length < 1280 + self.w_s + self.c_s:
            self.forceB = (self.P * train_x + self.P * (train_x + self.w_s) + self.P * (train_x + self.w_s + self.c_s) +
                           self.P * (train_x + 2 * self.w_s + self.c_s)) / self.B
            self.forceA = self.P * 4 - self.forceB
        elif train_x + self.train_length < 1280 + 2 * self.w_s + self.c_s:
            self.forceB = (self.P * train_x + self.P * (train_x + self.w_s) + self.P * (train_x + self.w_s + self.c_s)
                           ) / self.B
            self.forceA = self.P * 3 - self.forceB
        elif train_x + self.train_length < 1280 + 2 * self.w_s + 2 * self.c_s:
            self.forceB = (self.P * train_x + self.P * (train_x + self.w_s)) / self.B
            self.forceA = self.P * 2 - self.forceB
        elif train_x + self.train_length < 1280 + 3 * self.w_s + 2 * self.c_s:
            self.forceB = (self.P * train_x) / self.B
            self.forceA = self.P - self.forceB

    def shear_force(self, x, train_x):
        """ Returns the shear force at x mm from support A in N when the train is train_x mm from support A"""
        self.reaction_forces(train_x)  # Reaction forces required for shear, fill them out first

        if x < self.A:
            return 0
        if x == self.B:  # 0 conditions to stop infinate moment recursion. If x is somehow less than 0, return 0.
                         # If x is at support b, treat shear as 0 to prevent issues with moment calculations
            return 0

        # For each subsequent if statement, the code checks if the train and subsequently fewer wheels are to the
        # left of support b
        # For each wheel left of support B, we need to take away the wheel's downwards force for
        # the section of the SFD to the right of it before the next wheel
        if x < train_x and x < self.B:
            return self.forceA
        if x < train_x + self.w_s and x < self.B:
            return self.forceA - self.P
        elif x < train_x + self.w_s + self.c_s and x < self.B:
            return self.forceA - 2 * self.P
        elif x < train_x + 2 * self.w_s + self.c_s and x < self.B:
            return self.forceA - 3 * self.P
        elif x < train_x + 2 * self.w_s + 2 * self.c_s and x < self.B:
            return self.forceA - 4 * self.P
        elif x < train_x + 3 * self.w_s + 2 * self.c_s and x < self.B:
            return self.forceA - 5 * self.P

        # To consider the SFD to the right of the support, we need to add the support's upwards force to the SFD
        # Same statements as above but added to reaction force B
        elif self.B < x < train_x:
            return self.forceA + self.forceB
        elif self.B < x < train_x + self.w_s:
            return self.forceA + self.forceB - self.P
        elif self.B < x < train_x + self.w_s + self.c_s:
            return self.forceA + self.forceB - 2 * self.P
        elif self.B < x < train_x + 2 * self.w_s + self.c_s:
            return self.forceA + self.forceB - 3 * self.P
        elif self.B < x < train_x + 2 * self.w_s + 2 * self.c_s:
            return self.forceA + self.forceB - 4 * self.P
        elif self.B < x < train_x + 3 * self.w_s + 2 * self.c_s:
            return self.forceA + self.forceB - 5 * self.P

        # The code should never get to the statement directly below this
        elif x < self.B:
            return -self.forceB
        # This only runs if the train is completely before support B. Therefore, the SFD to the right of B is always 0
        elif x >= self.B:
            return 0


    def moment(self, x, train_x, recursive_def):
        """ Returns the moment at x mm from support A in Nmm when the train is train_x mm from support A"""
        """ Recursive_def is the recursive definition: helps manage how many recursive calls there are
        It is basically delta x for a manual integration"""
        shear_loc = self.shear_force(x, train_x)  # Local shear force for current x and train position

        if x <= 0: # If x is less than or equal to 0, moment is always 0 (free end, therefore net moment = 0)
            return 0

        # Check the lookup table of moments. If we already know the moment at the x we're looking for,
        # don't recursively find it again
        elif (x - recursive_def) in self.BMD_steven_is_the_best_ta:
            # Manual integration using recursion with respect to x, and dx is recursive_def. Takes the previous
            # height of the moment diagram and adds the slope (shear) times dX
            return shear_loc * recursive_def + self.BMD_steven_is_the_best_ta[x - recursive_def]
        else:
            # If not in the lookup table, find moment recursively. It is stored in the lookup table later
            return shear_loc * recursive_def + self.moment(x - recursive_def, train_x, recursive_def)

    def plot_diagrams(self):
        """ Plots the maximum and minimum shear and bending moment diagrams"""
        SFD_max = []  # To store the max and min moments for every distance x from support A (x = index)
        BMD_max = []
        SFD_min = []
        BMD_min = []

        sfd_big = []  # An array of arrays that stores the SFD and BMD for every train position. sfd_big[0] is the
        # SFD for when the train is at 0 mm from support A. This is required because a lookup table cannot be used
        # when the maximum value is found for every train position for a given x: otherwise would take too much
        # runtime
        bmd_big = []

        for train_x in range(1280):  # Loop through train positions
            shear = []  # Prep to store the SFD for cur train position (train_x)
            moment = []
            for x in range(1280):
                shear.append(self.shear_force(x, train_x)) # Append shear and moment to SFD for current train position
                moment.append(self.moment(x, train_x, 1))
                self.BMD_steven_is_the_best_ta[x] = moment[x] # Store the found moment in the lookup table

            sfd_big.append(shear)  # Append the SFD and BMD to the list of SFDs and BMDs
            bmd_big.append(moment)
            self.BMD_steven_is_the_best_ta.clear()  # When the train moves all BMD values need to be recalculated


        # loop through all x's and train positions this time in the inverse order, finding the max/min shear for
        # every x in all the SFDs. Append them to the maximum and minimum arrays
        for x in range(1280):
            big_shear = 0
            smol_shear = 0
            big_moment = 0
            smol_moment = 0
            for train_x in range(1280):
                shear = sfd_big[train_x][x]
                moment = bmd_big[train_x][x]
                if shear > big_shear:
                    big_shear = shear
                if shear < smol_shear:
                    smol_shear = shear
                if moment > big_moment:
                    big_moment = moment
                if moment < smol_moment:
                    smol_moment = moment
            SFD_max.append(big_shear)
            SFD_min.append(smol_shear)
            BMD_max.append(big_moment)
            BMD_min.append(smol_moment)

        # using the variable axs for multiple Axes
        fig, axs1 = plt.subplots(1, 2)

        # using tuple unpacking for multiple Axes
        axs1[0].plot(SFD_max)
        axs1[0].plot(SFD_min)
        axs1[0].set_xlabel("Distance from support 1 (mm)")
        axs1[0].set_ylabel("Shear force (N)")
        axs1[1].plot(BMD_max)
        axs1[1].plot(BMD_min)
        axs1[1].invert_yaxis()
        axs1[1].set_xlabel("Distance from support 1 (mm)")
        axs1[1].set_ylabel("Moment (Nmm)")
        axs1[0].legend(["Max Shear Force", "Min Shear Force"])
        axs1[1].legend(["Max Moment", "Min Moment"])
        plt.show()

        # Return the arrays of maxs and mins so they don't have to be recalculated for force analysis
        return SFD_max, SFD_min, BMD_max, BMD_min


class Diagrams:
    """ Generates the composite functions representing both moment diagrams based on a given P
    Very similar to for the train, except much simpler because the maxs and mins don't have to be considered because
    there is only one loading position
    """

    def __init__(self, P):
        self.A = 0  # Sup A distance
        self.P1 = 550  # load 1 distance
        self.B = self.P1 + 510  # Sup B distance
        self.P2 = self.B + 190  # Load 2 distance
        self.P = P  # P is the passed in load

        # Calculate reaction forces (simpler because only 2 point loads)
        self.Fby = (self.P2 * P + (self.P1 * P)) / self.B
        self.Fay = P * 2 - self.Fby

    def shear_force(self, x):
        """ Returns the shear force at x mm from support A in N
        Check which 'zone' the x is in and return the appropriate combination of reaction forces and loads"""
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
        """ Returns the moment at x mm from support A in Nmm
        Check which 'zone' the x is in and return the appropriate combination of reaction forces and loads"""
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
    """ Takes in a list of the rectangles of a split-up cross section in the form (B, H, d_bottom (y), x_pos)
    All 3 in mm
    d_bottom represents the distance between lower edge of the rectangle and the bottom of the cross-section
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
        """ Calculates and returns the centroid of a given cross-section
        Method: Sum all the areas times distance from bottom to centroid of area and find total area, divide
        when done with all rectangles"""
        numerator = 0  # Area times distance of rectangle's centroid to bottom
        area_total = 0
        for rect in section:
            h = abs(rect.h)  # In case a negative rectangle is passed in, deal with it without messing up centroid
            area_section = rect.w * h
            numerator += area_section * (abs(rect.y) + (h / 2))  # A * d
            area_total += area_section  # A tot

        return numerator / area_total

    def get_i_section(self):
        """ Calculates and returns the second moment of area of a given cross-section"""
        I = 0
        for rect in self.sections:
            h = abs(rect.h)
            # For every rectangle in the crossection, add I + A * d ^ 2
            I += self.i_rect(rect.w, h) + (rect.w * h) * (abs(rect.y + (h / 2) - self.centroid) ** 2)
        return I

    def calculate_q_section(self):
        """ Calculates and returns the first moment of area of a given cross-section"""
        max_height = 0  # Find max height to find area from
        for rect in self.sections:
            if rect.h + rect.y > max_height:
                max_height = rect.h + rect.y
        precision = C.PRECISION  # How many times to iterate through the height of the shape

        # Loop through height of shape C.PRECISION times. Finds Q at each height so the max can be found accurately
        for i in range(1, int(max_height * precision)):
            A = 0
            rects = []
            # Find Q at height i from the bottom Process: For each rectangle, find it's area above the centroid and
            # it's centroid, multiply them and add it to the total

            # Finds the rectangles above the point of interest. Slices parts of the crossection (webs) if need be
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
            d = abs(self.centroid - centroid_Q_area)  # d from centroid of shape above point of interest
            for rect in rects:
                A += rect.w * rect.h  # Total area

            self.Q.append(A * d)

    def get_Q(self):
        # Returns the array of the Q's at all heights of the shape. Useful for finding glue failure
        return self.Q

    def get_max_Q(self):
        # Returns the singular max value of Q and it's height above the bottom, for finding Matboard shear failure
        maximum = max(self.Q)
        index = self.Q.index(maximum)
        return maximum, index

    def get_separated_deck(self):
        """ Returns Rect objects that represent the geometrically split up deck, and which case of plate they are
                                                    The case is inversed: n:1 = case 2 and n:2 = case 1
        Is a horrible mess.
        No idea how general this is, but it works for the pi beam and box tube
        General method: loop through all cross sections and find the horizontal and vertical overlaps of the rectangles
        Only split the deck into multiple plates horizontally (flanges and middle), webs are 1 consistent plate
        Where 2 rectangles are touching, the projected width of one rectangle onto the other is cut out of the larger
        rectangle and stored in an array of "joints". The rest are stored as plate_rects
        Case number is found by taking each plate and finding out how many joints it's touching"""
        contact_areas = {}
        contact_areas_y = {}

        # For every deck in the shape (deck or horizontal plate in general), append the overlap and the y to an array
        for n, rect in enumerate(self.sections[:-1]):
            if rect.ID == C.DECK_ID:
                for other_rect in list(self.sections[:n] + self.sections[n + 1:]):
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
                        if round(plate_rects[c].w, 2) == round(rect_joint.w, 2) and round(plate_rects[c].x, 2) == round(
                                rect_joint.x, 2):
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
        # Takes in the cross section and splits the webs about the centroid (returns array of webs above and below centroid)
        sliced_webs = []
        case_n = []
        for rect in self.sections:
            if rect.ID == 'WEB':
                sliced_web = Rectangle(rect.w, rect.d_bottom + rect.height - self.centroid, rect.d_bottom, rect.x_pos)
                sliced_webs.append(sliced_web)
                case_n.append(3)  # Case is always 3 for webs

        return sliced_webs, case_n


class BridgeSolver:

    def __init__(self, cross_sections, SFD, BMD):
        self.cross_sections = cross_sections  # The cross section for every x of the bridge (index = x = dist from left)
        self.Is = []
        self.centroids = []
        self.Qs = []
        self.QsAllY = []  # Array of arrays that store the Q's for every height for every cross section
        self.sliced_decks = []
        self.sliced_webs = []
        # Calculates above values given the cross section. Uses the crosssectionsolver class
        self.solve_section_properties()
        self.V_fail = []  # Shear failure for each X

        # OLD: ARRAYS BELOW NOW STORE THE SHEAR FORCE AND MOMENT CAUSING FAILURE, NOT LOADS
        self.P_fail_C = []  # Load causing compression failure for every cross section
        self.P_fail_T = []  # Load causing tension failure for every cross section
        self.V_fail_MAT = []  # Load causing shear failure of the matboard for every cross section
        self.V_fail_glue = []  # Load causing shear failure of the glue for every cross section
        self.V_fail_MAT_buckling = [] # Load causing shear buckling failure for every cross section
        self.P_fail_PLATE = []  # Load causing plate buckling for every cross section
        self.SFD = SFD
        self.BMD = BMD
        self.CMD = self.get_curvature_diagram()

    def get_curvature_diagram(self):
        """Return the curvature diagram for a bridge (self)"""
        curvs = []
        # Curvature diagram = BMD / EI
        for x in range(C.BRIDGE_LENGTH):
            curvs.append(self.BMD[x] / (self.Is[x] * C.E))
        return curvs

    def solve_section_properties(self):
        """Gets all the section properties (I, Q, centroid, plates, etc) for every cross section in the bridge"""
        for cross_section in self.cross_sections:
            solver = CrossSectionSolver(cross_section)
            self.Is.append(solver.get_i_section())
            self.Qs.append(solver.get_max_Q())
            self.QsAllY.append(solver.get_Q())
            self.centroids.append(solver.centroid)
            self.sliced_decks.append(solver.get_separated_deck())
            self.sliced_webs.append(solver.get_sliced_webs())

    def flex_failure(self):
        """Calculates flexural failure shear forces for the bridge, both tension and compression"""

        # For every cross section in the bridge (1280 mm)
        for x in range(C.BRIDGE_LENGTH):
            if self.BMD[x] > 0:  # If bottom in tension
                max_height = 0  # Find the max height (greatest shear force present)
                for rect in self.cross_sections[x]:
                    if rect.h + rect.y > max_height:
                        max_height = rect.h + rect.y

                # Use compression and tension max stress in the equation stress = My/I to find max moment
                # Cap the moment at max_force to stop the graph from approaching infinity
                self.P_fail_C.append(min(
                    ((C.SigC * self.Is[x]) / (max_height - self.centroids[x])), C.MAX_FORCE))
                self.P_fail_T.append(
                    min(C.MAX_FORCE, ((C.SigT * self.Is[x]) / abs(self.centroids[x]))))
            elif self.BMD[x] < 0:  # Else if top in tension
                # Same process just treat the tension and compression opposite (y is diff sign)
                max_height = 0
                for rect in self.cross_sections[x]:
                    if rect.h + rect.y > max_height:
                        max_height = rect.h + rect.y
                self.P_fail_C.append(min(C.MAX_FORCE, ((C.SigC * self.Is[x]) / abs(
                    self.centroids[x]))))
                self.P_fail_T.append(min(C.MAX_FORCE,
                                         ((C.SigT * self.Is[x]) / abs(max_height - self.centroids[x]))))
            else: # If the moment diagram is 0, technically the max moment is infinity. We cap it at a million
                self.P_fail_C.append(1000000)
                self.P_fail_T.append(1000000)

    def shear_failure(self):
        # Calculate the shear failure of the matboard for every cross section of the bridge. Append values to V_fail_MAT
        for x in range(C.BRIDGE_LENGTH):
            q, y = self.Qs[x]
            t = 2 * 1.27  # CALCULATE IN CROSS-SECTION SOLVER
            try:  # If Q is 0 (edges), shear failure is technically infinite (divide by 0). Therefore, the only
                # reasonable thing to do is catch the error and append an unreasonably large value because answers
                # are more important than the process
                self.V_fail_MAT.append(((C.TauM * self.Is[x] * t) / q))
            except ZeroDivisionError:
                self.V_fail_MAT.append(1000000)

    def glue_fail(self):
        """Calculate the glue failure for any contact point between 2 rectangles that is on the horizontal axis"""
        # Sweep though height
        # Calculate overlapping rectangles
        # Note: Max tab width < a (spacing of diaphragms) Otherwise diaphragms are ignored
        default_len = len(self.cross_sections[0])
        max_tab_width = 30
        tab_counter = []

        # For every cross section, find the overlaps between all rectangles if they exist and store them to
        # contact_areas. Contact areas is a dictionary that stores overlaps on the same height above the bottom,
        # because they can be treated as one larger joint
        for x in range(C.BRIDGE_LENGTH):
            contact_areas = {}
            for i, rect in enumerate(self.cross_sections[x][:-1]):
                for other_rects in list(self.cross_sections[x][:i] + self.cross_sections[x][i + 1:]):
                    # Compare every deck to every other rectangle for contacting surfaces
                    # Assume contacting surfaces are all glued
                    if rect.ID == C.DECK_ID:
                        contact_area, y_glue = rect.is_touching(other_rects)
                        if contact_area and y_glue:
                            if y_glue in contact_areas.keys():
                                contact_areas[y_glue].append(contact_area)
                            else:
                                contact_areas[y_glue] = [contact_area]

            # Need to store all the glue failures for each cross section (if they are on diff heights) and find the
            # minimum. Bridge will fail when minimum is reached so the other can be discarded
            v_fail_glue_section = []
            min_sum = 100000000000000000000000000000000000000  # Need to start with the largest value so nothing is ignored
            y_smallest = 0

            # For every height of contact areas in the cross section, find b (min_sum) and the smallest Q

            for y, contact_area in contact_areas.items():
                if sum(contact_area) < min_sum:
                    min_sum = sum(contact_area)
                    y_smallest = y
                min_Q = self.QsAllY[x][int(y * C.PRECISION) - 1]
                v_fail = (C.TauG * self.Is[x] * sum(contact_area)) / min_Q  # Tau_max * I * b / Q = Max shear
                if y != C.NO_GLUE:  # If there is an exception and there is no glue on a horizontal tab, given this
                    # tag, it will be ignored
                    v_fail_glue_section.append(v_fail)

            try:  # Because who doesn't love a good try catch? What's it catching? Nobody knows, but he isn't
                # bothering anybody anyways
                self.V_fail_glue.append(min(v_fail_glue_section))
            except ZeroDivisionError:
                self.V_fail_glue.append(1000000)

    def plate_buckling(self):
        """Calculate the plate buckling moment for every given cross section"""
        for x in range(C.BRIDGE_LENGTH):
            # Large initial failure stress, don't ignore any possible stress (even if large)
            sigma_crit = 1000000000000000000
            sigma_crits = []  # Need to store every stress for each plate for compairison of minimum later
            for single_layer in self.sliced_decks[x]: # For height of plate in sliced_decks
                for n in range(len(single_layer)):  # For every plate in each height of sliced_decks
                    sliced_deck = single_layer[n]
                    for i in range(len(sliced_deck)):  # Same as above
                        rect = self.sliced_decks[x][0][n][i]  # The plate
                        case_n = self.sliced_decks[x][1][n][i]  # Case number of the plate (remember - reversed)
                        if abs(rect.w) < 100 and rect.w > 0:  # Plates should be smaller than the deck
                            if case_n == 2:
                                up_or_down = rect.y - self.centroids[x]  # This is b (width of plate)
                                # Append this plate's buckling stress to the array. Take min later
                                sigma_crits.append([self.case_1(rect.h, rect.w), up_or_down])

                            elif case_n == 1:
                                up_or_down = rect.y - self.centroids[x]  # This is b
                                sigma_crits.append([self.case_2(rect.h, rect.w), up_or_down])

            # Loop through all sliced webs and consider their critical stresses
            for i in range(len(self.sliced_webs[x][0])):
                rect = self.sliced_webs[x][0][i]
                case_n = self.sliced_webs[x][1][i]
                if case_n == 3:  # Should always be 3, but this is a final check
                    # The critical stress is equal to the min of the previous minimum or the current critical stress
                    sigma_crit = min(self.case_3(rect.w, rect.h), sigma_crit)

            # Compare the vertical plates critical stress to the web's critical stress
            # Need to consider which side is in compression for sign of critical stress
            if self.BMD[x] > 0:
                for sg in sigma_crits:
                    if sg[1] > 0:
                        sigma_crit = min(sg[0], sigma_crit)
            elif self.BMD[x] < 0:
                for sg in sigma_crits:
                    if sg[1] < 0:
                        sigma_crit = min(sg[0], sigma_crit)

            # Actually append the mimimum moment by taking the smallest critical stress and plugging into stress = MY/I
            if self.BMD[x] > 0:
                max_height = 0
                for rect in self.cross_sections[x]:
                    if rect.h + rect.y > max_height:
                        max_height = rect.h + rect.y # Need the largest Y for smallest moment to consider
                P_fail = min((
                    ((sigma_crit * self.Is[x]) / (max_height - self.centroids[x])),
                    C.MAX_FORCE))
            elif self.BMD[x] < 0:
                max_height = 0
                for rect in self.cross_sections[x]:
                    if rect.h + rect.y > max_height:
                        max_height = rect.h + rect.y
                P_fail = min(C.MAX_FORCE, ((sigma_crit * self.Is[x]) / abs(
                    self.centroids[x])))
            else:
                P_fail = C.MAX_FORCE # If moment is 0, failure approaches infinity and that isn't pretty on graphs
            self.P_fail_PLATE.append(P_fail) # Append the calculated minimum failure moment

    @staticmethod
    def case_1(t, b):
        """Returns critical stress for 2 ends constrained plate"""
        return ((4 * (math.pi ** 2) * C.E) / (12 * (1 - C.mu ** 2))) * ((t / b) ** 2)

    @staticmethod
    def case_2(t, b):
        """Returns critical stress for one end constrained plate"""
        return ((0.425 * (math.pi ** 2) * C.E) / (12 * (1 - C.mu ** 2))) * ((t / b) ** 2)

    @staticmethod
    def case_3(t, b):
        """Returns critical stress for a plate where stress varies linearly"""
        return ((6 * (math.pi ** 2) * C.E) / (12 * (1 - C.mu ** 2))) * ((t / b) ** 2)

    @staticmethod
    def get_shear_buckling(t, h, a):
        """Returns critical shear for plates that experience shear buckling"""
        return ((5 * (math.pi ** 2) * C.E) / (12 * (1 - C.mu ** 2))) * ((t / h) ** 2 + (t / a) ** 2)

    def shear_buckling(self):
        """Find the failure shear for a bridge in shear buckling"""
        for x in range(C.BRIDGE_LENGTH):
            h_web = 0
            for rect in self.cross_sections[x]:
                h_web = max(rect.h, h_web)  # Find the max web height
            a = 550  # Diaphragm spacing. Manually insert here
            # Find the critical shear for buckling failure
            tau = self.get_shear_buckling(C.DIAPHRAGM_THICKNESS, h_web, a)
            q, y = self.Qs[x]
            t = C.MATBOARD_THICKNESS
            try:  # A nice try, but seems to have shear'd my exception away
                self.V_fail_MAT_buckling.append(((tau * self.Is[x] * t * 2) / q))
            except ZeroDivisionError:
                self.V_fail_MAT_buckling.append(1000000)  # If Q is 0, failure shear is infinite so append large value

    def midspan_deflection(self):
        """Calculte deflection of the midspan under 2 point loads"""
        max_curvature = max(self.CMD)
        min_curvature = min(self.CMD)

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
        comp_FOS = 1000000  # FOS initially large so all FOS are considered no matter how large
        tens_FOS = 1000000
        sMAT_FOS = 1000000
        sGLUE_FOS = 1000000
        sbMATH_FOS = 1000000
        PLATE_FOS = 1000000
        for x in range(1280):
            # Don't find FOS if either diagram is 0, throws an error and is infinity anyways
            if self.BMD[x] != 0:
                comp_FOS = min(comp_FOS, abs(self.P_fail_C[x] / self.BMD[x]))
                tens_FOS = min(tens_FOS, abs(self.P_fail_T[x] / self.BMD[x]))
                PLATE_FOS = min(PLATE_FOS, abs(self.P_fail_PLATE[x] / self.BMD[x]))
            if self.SFD[x] != 0:
                sMAT_FOS = min(sMAT_FOS, abs(self.V_fail_MAT[x] / self.SFD[x]))
                sGLUE_FOS = min(sGLUE_FOS, abs(self.V_fail_glue[x] / self.SFD[x]))
                sbMATH_FOS = min(sbMATH_FOS, abs(self.V_fail_MAT_buckling[x] / self.SFD[x]))

        min_any = min(comp_FOS, tens_FOS, sMAT_FOS, sGLUE_FOS, sbMATH_FOS, PLATE_FOS)
        # Find minimum FOS and print
        print(f"Factor of safety against train: {round(min_any, 3)}")

    def plot(self):
        """Plot all failure values on the SFD and BMD"""
        min_P_comp = min(self.P_fail_C)
        min_P_tens = min(self.P_fail_T)
        min_P_sMAT = min(self.V_fail_MAT)
        min_P_sGLUE = min(self.V_fail_glue)
        min_P_sbMATH = min(self.V_fail_MAT_buckling)
        min_P_PLATE = min(self.P_fail_PLATE)

        # Print the minimum values for easy comprehension
        print(f"Min Moment Fail Compression (flexural): {min_P_comp} Nmm, at x={self.P_fail_C.index(min_P_comp)}")
        print(f"Min Moment Fail Tension (flexural): {min_P_tens} Nmm, at x={self.P_fail_T.index(min_P_tens)}")
        print(f"Min Force Fail Matboard (shear): {min_P_sMAT} N, at x={self.V_fail_MAT.index(min_P_sMAT)}")
        print(f"Min Force Fail Glue (shear): {min_P_sGLUE} N, at x={self.V_fail_glue.index(min_P_sGLUE)}")
        print(f"Min Force Fail Shear Buckling: {min_P_sbMATH} N, at x={self.V_fail_MAT_buckling.index(min_P_sbMATH)}")
        print(f"Min Force Fail Plate Buckling: {min_P_PLATE} Nmm, at x={self.P_fail_PLATE.index(min_P_PLATE)}")

        # Print and calculate FOS
        self.FOS()
        fig, axs = plt.subplots(1, 2)

        axs[0].plot(self.SFD)
        axs[0].plot(self.V_fail_MAT)
        axs[0].plot(self.V_fail_glue)
        axs[0].plot(self.V_fail_MAT_buckling)
        axs[0].legend(["Shear Force", "Matboard Shear", "Glue Shear", "Shear Buckle"])
        axs[0].set_xlabel("Distance from support 1 [mm]")
        axs[0].set_ylabel("Shear Force (N)")

        axs[1].plot(self.BMD)
        axs[1].plot(self.P_fail_C)
        axs[1].plot(self.P_fail_T)
        axs[1].plot(self.P_fail_PLATE)
        axs[1].legend(["Bending Moment", "Flexural Failure (comp)", "Flexural Failure (tens)", "Plate Buckling"])
        axs[1].set_xlabel("Distance from support 1 [mm]")
        axs[1].set_ylabel("Moment [Nmm]")
        plt.show()


class Arch:

    # Help define cross sections that vary over the span of the bridge
    @staticmethod
    def under_arch(x):
        y = abs(0.2 * (x - 549)) - 100
        if y < 0:
            return abs(y)
        else:
            return 0

    @staticmethod
    def over_arch(x):
        #  y = -0.0017 * (x - 788) * (x - 1280)
        if x < 1060:
            y = -abs(0.3 * (x - 1059)) + 150
        else:
            y = -abs(0.2 * (x - 1059)) + 150
        if y > 0:
            return y
        else:
            return 0

    @staticmethod
    def pi_beam_remover(x):
        y = abs(0.2 * (x - 1280)) - 40
        if y < 0 and x > 1059:
            return abs(y)
        else:
            return 0


def generate_cross_sections(arch):
    """Generate the cross sections for the entire bridge as an array"""
    bridge_length = 1280
    cross_sections = []
    deck = [100, 1.27, 0]

    # Apend the cross sections that can be varied over the length of the bridge
    for x in range(bridge_length):
        y_upper = arch.over_arch(x)
        y_under = arch.under_arch(x)
        pi_remove = arch.pi_beam_remover(x)

        '''
        # DESIGN 0
        t = 1.27
        deck = [100, t, 75 - t, 0]  # top is 75+t
        tab_1 = [10, t, (75 - 2 * t), 10 + t]
        tab_2 = [10, t, (75 - 2 * t), (90 - t - 10)]

        cross_sections.append([Rectangle(deck[0], 1.27, deck[2], deck[3], C.DECK_ID),  # Deck
                               Rectangle(t, (deck[2] - t), t, 10, C.WEB_ID),  # Web1
                               Rectangle(t, (deck[2] - t), t, (90 - t), C.WEB_ID),  # Web2
                               Rectangle(80, t, 0, 10 + t, C.DECK_ID),  # bottom
                               Rectangle(tab_1[0], tab_1[1], tab_1[2], tab_1[3], C.GLUE_TAB_ID),  # Glue_tab1
                               Rectangle(tab_2[0], tab_2[1], tab_2[2], tab_2[3], C.GLUE_TAB_ID)])  # Glue tab 2
        '''

        # OUR DESIGN
        # Each cross section is an array of rectangles
        deck = [100, 1.27, 75, 0]
        arch_rect = [1.27, 75, 0, 10 - 1.27]
        arch_rect_2 = [1.27, 75, 0, (90)]
        tab_1 = [10, 1.27, (75 - 1.27), 10]
        tab_2 = [10, 1.27, (75 - 1.27), (90 - 10)]
        cross_sections.append([Rectangle(deck[0], deck[1], deck[2] + y_under - pi_remove, deck[3], C.DECK_ID),
                               Rectangle(arch_rect[0], arch_rect[1] + y_under - pi_remove, arch_rect[2], arch_rect[3],
                                         C.WEB_ID),
                               Rectangle(arch_rect_2[0], arch_rect_2[1] + y_under - pi_remove, arch_rect_2[2],
                                         arch_rect_2[3], C.WEB_ID),
                               Rectangle(tab_1[0], tab_1[1], tab_1[2] + y_under - pi_remove, tab_1[3], C.GLUE_TAB_ID),
                               Rectangle(tab_2[0], tab_2[1], tab_2[2] + y_under - pi_remove, tab_2[3], C.GLUE_TAB_ID),
                               Rectangle(arch_rect[0], y_upper, deck[2] + 1.27 + y_under - pi_remove, 0),
                               Rectangle(arch_rect[0], y_upper, deck[2] + 1.27 + y_under - pi_remove, 100 - 1.27),
                               Rectangle(tab_1[0] - 1.27, tab_1[1], deck[2] + 1.27 + y_under - pi_remove, 1.27,
                                         C.GLUE_TAB_ID),
                               Rectangle(tab_2[0] - 1.27, tab_2[1], deck[2] + 1.27 + y_under - pi_remove,
                                         100 - 1.27 - 10, C.GLUE_TAB_ID),
                               ])

        """
        # Other considered designs:
        
        deck = [120, 1.27, 50, 0]
        arch_rect = [1.27, 50, 0, 10-1.27]
        arch_rect_2 = [1.27, 50, 0, (110)]
        tab_1 = [10, 1.27, (50 - 1.27), 10]
        tab_2 = [10, 1.27, (50 - 1.27), (110 - 10 - 1.27)]
        
        cross_sections.append([Rectangle(100, 2.54, 121.27, 0, C.DECK_ID),
                              Rectangle(1.27, 120, 1.27, 15, C.WEB_ID),
                              Rectangle(1.27, 120, 1.27, 15+68, C.WEB_ID),
                            Rectangle(70, 1.27, 0, 15, C.DECK_ID)

        ])
            cross_sections.append([Rectangle(deck[0], deck[1], deck[2]+y_under, deck[3], C.DECK_ID),
                                   Rectangle(arch_rect[0], arch_rect[1]+y_under, arch_rect[2], arch_rect[3], C.WEB_ID),
                                   Rectangle(arch_rect_2[0], arch_rect_2[1]+y_under, arch_rect_2[2], arch_rect_2[3], C.WEB_ID),
                                   Rectangle(tab_1[0], tab_1[1], tab_1[2]+y_under, tab_1[3], C.GLUE_TAB_ID),
                                   Rectangle(tab_2[0], tab_2[1], tab_2[2]+y_under, tab_2[3], C.GLUE_TAB_ID)
                                   ])
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


def solve_loads():
    """Call all the required functions to print and plot the desired values under load case 2"""
    diagrams = Diagrams(C.P)
    diagrams.plot_diagrams(0)
    cross_sections = generate_cross_sections(Arch())
    SFD = []
    BMD = []


    # Uncomment below lines to visualize the cross section and elevation view of the bridge
    # import c_s_visualizer
    # cs = c_s_visualizer.DrawCrossSection(cross_sections, None)
    # cs.draw_animation()
    # c_s_visualizer.DrawElevation(cross_sections).draw(549, None)

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


def solve_train():
    """Call all the required functions to print and plot the desired values under load case 1 (train)"""
    diagrams = Diagrams_case1()
    SFD_max, SFD_min, BMD_max, BMD_min = diagrams.plot_diagrams()
    cross_sections = generate_cross_sections(Arch())

    # Uncomment below lines to visualize the cross section and elevation view of the bridge
    # import c_s_visualizer
    # cs = c_s_visualizer.DrawCrossSection(cross_sections, None)
    # cs.draw_animation()
    # c_s_visualizer.DrawElevation(cross_sections).draw(549, None)

    bridge_solver_max = BridgeSolver(cross_sections, SFD_max, BMD_max)
    bridge_solver_max.flex_failure()
    bridge_solver_max.shear_failure()
    bridge_solver_max.glue_fail()
    bridge_solver_max.shear_buckling()
    bridge_solver_max.plate_buckling()
    # bridge_solver.midspan_deflection()
    bridge_solver_max.plot()

    bridge_solver_min = BridgeSolver(cross_sections, SFD_min, BMD_min)
    bridge_solver_min.flex_failure()
    bridge_solver_min.shear_failure()
    bridge_solver_min.glue_fail()
    bridge_solver_min.shear_buckling()
    bridge_solver_min.plate_buckling()
    # bridge_solver.midspan_deflection()
    bridge_solver_min.plot()


if __name__ == "__main__":
    solve_train()
    # solve_loads()
