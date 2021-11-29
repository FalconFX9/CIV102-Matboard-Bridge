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
            n_rect = [rect.x, rect.y, rect.w, rect.h]
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
        n = 500
        run = True
        from main import Rectangle
        while run:
            self.window.fill((0, 0, 0))
            rects = self.rects[n]
            for rect in rects:
                n_rect = Rectangle(rect.w, rect.h, rect.y, rect.x)
                n_rect.x += (self.window.get_size()[0]/2)
                n_rect.y += (self.window.get_size()[1]/2)
                pygame.draw.rect(self.window, (255, 255, 255), [n_rect.x, n_rect.y, n_rect.w, n_rect.h])

            self.window.blit(pygame.transform.rotate(self.window, 180), (0, 0))
            pygame.display.flip()
            pygame.event.pump()
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    run = False
            n += 1
            print(n)
            if n >= 1280:
                n = 0


class DrawElevation:

    def __init__(self, cross_sections):
        self.window = pygame.display.set_mode((1280, 700))
        self.cross_sections = cross_sections

    def draw(self, failure_location, arch):
        self.window.fill((0, 0, 0))
        for x in range(1279, -1, -1):
            h_max = 0
            for rect in self.cross_sections[1279-x]:
                h_max = max(h_max, rect.y + rect.h)
            pygame.draw.rect(self.window, (255, 255, 255), [x, self.window.get_size()[1]/2-arch.under_arch(1279-x)+arch.pi_beam_remover(1279-x), 1, h_max])

        pygame.draw.rect(self.window, (255, 0, 0), [1265, self.window.get_size()[1]/2, 15, 5])
        pygame.draw.rect(self.window, (255, 0, 0), [1280-1075, self.window.get_size()[1] / 2, 30, 5])
        pygame.draw.rect(self.window, (255, 0, 0), [1280-failure_location, self.window.get_size()[1] / 2, 1, 30])
        pygame.draw.rect(self.window, (0, 255, 0), [1280-552, self.window.get_size()[1]/2+100, 4, 50])
        pygame.draw.rect(self.window, (0, 255, 0), [1280 - 1252, self.window.get_size()[1] / 2 + 100, 4, 50])
        clock = pygame.time.Clock()
        self.window.blit(pygame.transform.rotate(self.window, 180), (0, 0))
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
