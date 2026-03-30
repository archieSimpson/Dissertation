import pygame

from config import SimulationConfig
from render_pygame import render
from simulation import ShepherdingSimulation


def main():
    cfg = SimulationConfig()

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Shepherding Simulation - Research Upgrade")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", cfg.font_size)

    sim = ShepherdingSimulation(cfg)

    running = True
    while running:
        clock.tick(cfg.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    sim.toggle_dog_mode()

        keys = pygame.key.get_pressed()
        sim.step(keys)
        render(screen, font, sim)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()