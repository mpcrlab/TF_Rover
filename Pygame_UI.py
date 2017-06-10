import pygame

class Pygame_UI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Dashboard')
        self.screen_size = [700, 480]
        self.screen = pygame.display.set_mode(self.screen_size)
        self.screen.fill((255,255,255))
        self.fontSize = 30
        self.font = pygame.font.SysFont(None, self.fontSize)
        self.clock = pygame.time.Clock()
        self.color = (0,0,0)

    def display_message(self, text, color, x, y):
        label = self.font.render(text, True, color)
        self.screen.blit(label, (x,y))
