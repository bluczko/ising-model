import numpy as np
import pygame as pg

from ising.model import IsingModel


pg.init()
resolution = (1024, 1024)
display = pg.display.set_mode(resolution, pg.RESIZABLE)

pg.display.set_caption("Model Isinga", "")
pg.key.set_repeat(450, 100)
running = True

stats = np.zeros(shape=(3, 10))
ising = IsingModel(256, 1.0)
i = 0

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.VIDEORESIZE:
            resolution = (event.w, event.h)
            display = pg.display.set_mode(resolution, pg.RESIZABLE)

    pressed = pg.key.get_pressed()
    update_caption = False

    if pressed[pg.K_PLUS] or pressed[pg.K_KP_PLUS]:
        ising.temperature = min((5.0, ising.temperature + 0.1))
        update_caption = True

    elif pressed[pg.K_MINUS] or pressed[pg.K_KP_MINUS]:
        ising.temperature = max((0.1, ising.temperature - 0.1))
        update_caption = True

    ising.step()

    i = (i + 1) % stats.shape[1]
    stats[0, i] = ising.temperature
    stats[1, i] = ising.magnetization
    stats[2, i] = ising.heat_capacity
    sm = stats.mean(axis=1)

    pg.display.set_caption(f"Model Isinga: T={sm[0]:.1f}, M={sm[1]:.6f}, H={sm[2]:.2f}", "")

    system_surface = pg.surfarray.make_surface(ising.system)
    system_surface = pg.transform.scale(system_surface, resolution)

    display.fill((0, 0, 0))
    display.blit(system_surface, dest=(0, 0))
    pg.display.update()

pg.quit()
