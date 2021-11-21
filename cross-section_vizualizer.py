# GUI to visualize the cross section at any depth, as well as the elevation view of the bridge (if I have the time to
# implement that)


import pygame


class DrawCrossSection:

    def __init__(self, rectangles, window):
        self.rects = rectangles
        self.window = window

    def draw(self):
        self.window.fill((0, 0, 0))
        for rect in self.rects:
            rect[1] *= -1
            rect[1] -= rect[3]
            print(rect)
            rect[0] += self.window.get_size()[0]/2 - 50
            rect[1] += self.window.get_size()[1]/2 - 65

            pygame.draw.rect(self.window, (255, 255, 255), rect)


class DrawElevation:

    def __init__(self, window):
        self.window = window

    def draw(self):
        self.window.fill((0, 0, 0))
        from main import Arch
        for x in range(1280):
            y1 = Arch.over_arch(x)
            y2 = Arch.under_arch(x)
            pygame.draw.rect(self.window, (255, 255, 255), (x - 1, (self.window.get_size()[1]/2), 1, -y2))
            pygame.draw.rect(self.window, (255, 255, 255), (x - 1, (self.window.get_size()[1] / 2 - y1), 1, y1))


if __name__ == "__main__":
    screen = pygame.display.set_mode((1280, 700))
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
