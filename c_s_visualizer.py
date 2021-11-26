# GUI to visualize the cross section at any depth, as well as the elevation view of the bridge (if I have the time to
# implement that)


import pygame


class DrawCrossSection:

    def __init__(self, rectangles, window):
        self.rects = rectangles
        self.window = pygame.display.set_mode((1280, 800))

    def draw_single(self, n):
        self.window.fill((0, 0, 0))
        for rect in self.rects[n]:
            n_rect = [rect[3], rect[2], rect[0], rect[1]]
            n_rect[1] *= -1
            n_rect[1] -= n_rect[3]
            n_rect[0] += self.window.get_size()[0]/2 - 50
            n_rect[1] += self.window.get_size()[1]/2 - 65

            pygame.draw.rect(self.window, (255, 255, 255), n_rect)
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            pygame.display.flip()
            pygame.event.pump()
            clock.tick(60)

    def draw_animation(self):
        clock = pygame.time.Clock()
        n = 0
        run = True
        while run:
            self.window.fill((0, 0, 0))
            rects = self.rects[n]
            for rect in rects:

                n_rect = [rect[3], rect[2], rect[0], rect[1]]

                n_rect[0] += self.window.get_size()[0] / 2 - 50
                if rects[0][2] > 0:
                    n_rect[1] *= -1
                    n_rect[1] += n_rect[3]
                    n_rect[1] += (self.window.get_size()[1] / 2 - 65) + rects[0][2]
                else:
                    n_rect[1] -= n_rect[3]
                    n_rect[1] += (self.window.get_size()[1] / 2 - 65)
                n_rect[3] = abs(n_rect[3])

                pygame.draw.rect(self.window, (255, 255, 255), n_rect)

            pygame.display.flip()
            pygame.event.pump()
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    run = False
            n += 1
            if n >= 1280:
                n = 0


class DrawElevation:

    def __init__(self):
        self.window = pygame.display.set_mode((1280, 700))

    def draw(self, failure_location, arch):
        self.window.fill((0, 0, 0))
        for x in range(1280):
            y1 = arch.over_arch(x)
            y2 = arch.under_arch(x)
            y3 = arch.right_support_under_arch(x)
            pygame.draw.rect(self.window, (255, 255, 255), (x - 1, (self.window.get_size()[1]/2), 1, -y2))
            pygame.draw.rect(self.window, (255, 255, 255), (x - 1, (self.window.get_size()[1] / 2 - y1), 1, y1))
            pygame.draw.rect(self.window, (255, 255, 255), (x - 1, (self.window.get_size()[1]/2), 1, -y3))
        pygame.draw.rect(self.window, (255, 0, 0), [0, self.window.get_size()[1]/2, 15, 5])
        pygame.draw.rect(self.window, (255, 0, 0), [1045, self.window.get_size()[1] / 2, 30, 5])
        pygame.draw.rect(self.window, (255, 0, 0), [failure_location, self.window.get_size()[1] / 2, 1, 30])
        clock = pygame.time.Clock()
        run = True
        while run:
            pygame.display.flip()
            pygame.event.pump()
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    run = False


if __name__ == "__main__":
    #screen = pygame.display.set_mode((1280, 700))
    rects = [[0, 100, 100, 2], [0, 0, 2, 100], [98, 0, 2, 100]]

    cross_section = DrawCrossSection(rects, screen)
    clock = pygame.time.Clock()
    #cross_section.draw()
    elevation = DrawElevation(screen)
    elevation.draw()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        pygame.display.flip()
        pygame.event.pump()
        clock.tick(60)
