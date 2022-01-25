import pygame
import numpy as np
from gym_achtung.envs.achtungdiekurve import AchtungDieKurve, AchtungPlayer
from gym import spaces

WINWIDTH = 480  # width of the program's window, in pixels
WINHEIGHT = 480  # height in pixels
TEXT_SPACING = 130
RADIUS = 2      # radius of the circles
PLAYERS = 1      # number of players
SKIP_PROBABILITY = 0
SPEED_CONSTANT = 2
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
P1COLOUR = RED
P2COLOUR = GREEN
P3COLOUR = BLUE
BG_COLOR = (25, 25, 25)
BEAM_SIGHT = 240
BEAM_MAX_ANGLE = 120
BEAM_STEP = 30
BEAMS = range(BEAM_MAX_ANGLE, -BEAM_MAX_ANGLE-BEAM_STEP, -BEAM_STEP)


# basically just holds onto all of them

class AchtungDieKurveAgainstBot(AchtungDieKurve):

    def __init__(self):
        self.width = WINWIDTH
        self.aiscore = 0
        self.humanscore = -1
        self.aiwon = False
        self.human_init()

        super().__init__()

    def _setup(self):
        """
        Setups up the pygame env, the display and game clock.
        """
        pygame.init()
        self.screen = pygame.display.set_mode(self.getScreenDims())
        pygame.display.set_caption('Achtung AI AI score: {} Human score: {}'.format(self.aiscore, self.humanscore))
        self.clock = pygame.time.Clock()

    def _step(self):

        self._handle_human_player_events()
        self.humanplayer.update()
        if self.collision(self.humanplayer.x, self.humanplayer.y, self.humanplayer.skip):
            self.aiwon = True
            self.reset()
        self.humanplayer.draw(self.screen)
        super()._step()

    def human_init(self):
        self.humanplayer = AchtungPlayer(GREEN, RADIUS)

    def _handle_human_player_events(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.humanplayer.angle -= 10
            if self.humanplayer.angle <= 0:
                self.humanplayer.angle += 360
        if keys[pygame.K_RIGHT]:
            self.humanplayer.angle += 10
            if self.humanplayer.angle >= 360:
                self.humanplayer.angle -= 360

    def reset(self):
        if self.aiwon:
            self.aiscore += 1
        else:
            self.humanscore += 1
        pygame.display.set_caption('Achtung AI - AI score: {} Human score: {}'.format(self.aiscore, self.humanscore))
        self.human_init()
        self.aiwon = False
        self.observation_space = spaces.Box(low=0, high=WINWIDTH, shape=(12,), dtype=np.uint8)
        self.last_action = []
        self.action = []
        self.previous_score = 0.0
        self.init()
        state = self.getGameState()
        return state
        #super().reset()


if __name__ == "__main__":

    pygame.init()
    game = AchtungDieKurveAgainstBot()
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        if game.game_over():
            game.init()

        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
