import os
import sys
import math
import time
import pygame
current_path = os.getcwd()
import pymunk as pm
from characters import Bird
from level import Level
import requests
from threading import Thread, Lock
from mykeypointdetector import MyDetector
import cv2
import paddlex as pdx
import copy

results = None
foresults = None

os.environ['CUDA_VISIBLE_DEVICES']='0'

lock = Lock()

# 梳理一下
# 两个手小于一定的距离的时候,为开始拉弓,即对应着在弹弓附近的点击事件
# 两个手的角度对应角度
# 两个手的宽度对应力度
# 两个手的手势对应着送开弓
# 那还有哪些事件没有写呢,暂定面板\重来按钮\下一关

pygame.init()
screen = pygame.display.set_mode((1200, 650))
redbird = pygame.image.load(
    "../resources/images/red-bird3.png").convert_alpha()
background2 = pygame.image.load(
    "../resources/images/background3.png").convert_alpha()
sling_image = pygame.image.load(
    "../resources/images/sling-3.png").convert_alpha()
full_sprite = pygame.image.load(
    "../resources/images/full-sprite.png").convert_alpha()
rect = pygame.Rect(181, 1050, 50, 50)
cropped = full_sprite.subsurface(rect).copy()
pig_image = pygame.transform.scale(cropped, (30, 30))
buttons = pygame.image.load(
    "../resources/images/selected-buttons.png").convert_alpha()
pig_happy = pygame.image.load(
    "../resources/images/pig_failed.png").convert_alpha()
stars = pygame.image.load(
    "../resources/images/stars-edited.png").convert_alpha()
rect = pygame.Rect(0, 0, 200, 200)
star1 = stars.subsurface(rect).copy()
rect = pygame.Rect(204, 0, 200, 200)
star2 = stars.subsurface(rect).copy()
rect = pygame.Rect(426, 0, 200, 200)
star3 = stars.subsurface(rect).copy()
rect = pygame.Rect(164, 10, 60, 60)
pause_button = buttons.subsurface(rect).copy()
rect = pygame.Rect(24, 4, 100, 100)
replay_button = buttons.subsurface(rect).copy()
rect = pygame.Rect(142, 365, 130, 100)
next_button = buttons.subsurface(rect).copy()
clock = pygame.time.Clock()
rect = pygame.Rect(18, 212, 100, 100)
play_button = buttons.subsurface(rect).copy()
running = True
# the base of the physics
space = pm.Space()
space.gravity = (0.0, -700.0)
pigs = []
birds = []
balls = []
polys = []
beams = []
columns = []
poly_points = []
ball_number = 0
polys_dict = {}
mouse_distance = 0
rope_lenght = 90
angle = 0
x_mouse = 0
y_mouse = 0
count = 0
mouse_pressed = False
t1 = 0
t2 = 0
tick_to_next_circle = 10
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
sling_x, sling_y = 135, 450
sling2_x, sling2_y = 160, 450
score = 0
game_state = 0
bird_path = []
counter = 0
restart_counter = False
bonus_score_once = True

level_failed_time = 0
level_next_time = 0
bold_font = pygame.font.SysFont("arial", 30, bold=True)
bold_font2 = pygame.font.SysFont("arial", 40, bold=True)
bold_font3 = pygame.font.SysFont("arial", 50, bold=True)
wall = False

# Static floor
static_body = pm.Body(body_type=pm.Body.STATIC)
static_lines = [pm.Segment(static_body, (0.0, 060.0), (1200.0, 060.0), 0.0)]
static_lines1 = [pm.Segment(static_body, (1200.0, 060.0), (1200.0, 800.0), 0.0)]
for line in static_lines:
    line.elasticity = 0.95
    line.friction = 1
    line.collision_type = 3
for line in static_lines1:
    line.elasticity = 0.95
    line.friction = 1
    line.collision_type = 3
space.add(static_body)
for line in static_lines:
    space.add(line)


def to_pygame(p):
    """Convert pymunk to pygame coordinates"""
    return int(p.x), int(-p.y+600)


def vector(p0, p1):
    """Return the vector of the points
    p0 = (xo,yo), p1 = (x1,y1)"""
    a = p1[0] - p0[0]
    b = p1[1] - p0[1]
    return (a, b)


def unit_vector(v):
    """Return the unit vector of the points
    v = (a,b)"""
    h = ((v[0]**2)+(v[1]**2))**0.5
    if h == 0:
        h = 0.000000000000001
    ua = v[0] / h
    ub = v[1] / h
    return (ua, ub)


def distance(xo, yo, x, y):
    """distance between points"""
    dx = x - xo
    dy = y - yo
    d = ((dx ** 2) + (dy ** 2)) ** 0.5
    return d


def load_music():
    """Load the music"""
    song1 = '../resources/sounds/angry-birds.ogg'
    pygame.mixer.music.load(song1)
    pygame.mixer.music.play(-1)


'''
鼠标到弹弓的向量
mouse_distance 是鼠标到弹弓的距离

'''

def sling_action():
    """Set up sling behavior"""
    global mouse_distance
    global rope_lenght
    global angle
    global x_mouse
    global y_mouse
    # Fixing bird to the sling rope
    # 鼠标到弹弓的向量
    v = vector((sling_x, sling_y), (x_mouse, y_mouse))
    uv = unit_vector(v)
    uv1 = uv[0]
    uv2 = uv[1]
    # mouse_distance 是鼠标到弹弓的距离
    mouse_distance = distance(sling_x, sling_y, x_mouse, y_mouse)
    # pu是一个固定的长度
    # rope_lenght = 90
    pu = (uv1*rope_lenght+sling_x, uv2*rope_lenght+sling_y)
    bigger_rope = 102
    x_redbird = x_mouse - 20
    y_redbird = y_mouse - 20
    if mouse_distance > rope_lenght:
        # 这个时候，画绳子是使用固定的bigger_rope的半径
        # 而画鸟是固定的rope_length的长度（）-20
        pux, puy = pu
        pux -= 20
        puy -= 20
        pul = pux, puy
        # screen.blit(redbird, pul)
        pu2 = (uv1*bigger_rope+sling_x, uv2*bigger_rope+sling_y)
        pygame.draw.line(screen, (0, 0, 0), (sling2_x, sling2_y), pu2, 5)
        # bird就画在固定的位置， 这个半径是固定的rope_length
        screen.blit(redbird, pul)
        pygame.draw.line(screen, (0, 0, 0), (sling_x, sling_y), pu2, 5)
    else:
        # 真实的鼠标半径+10
        mouse_distance += 10
        pu3 = (uv1*mouse_distance+sling_x, uv2*mouse_distance+sling_y)
        pygame.draw.line(screen, (0, 0, 0), (sling2_x, sling2_y), pu3, 5)
        screen.blit(redbird, (x_redbird, y_redbird))
        pygame.draw.line(screen, (0, 0, 0), (sling_x, sling_y), pu3, 5)
    # Angle of impulse
    dy = y_mouse - sling_y
    dx = x_mouse - sling_x
    if dx == 0:
        dx = 0.00000000000001
    angle = math.atan((float(dy))/dx)


def sling_action_pose(sl, sr):
    """Set up sling behavior"""
    global mouse_distance
    global rope_lenght
    global angle
    global x_mouse
    global y_mouse
    # Fixing bird to the sling rope
    # 鼠标到弹弓的向量
    v = vector(sr,sl)
    uv = unit_vector(v)
    uv1 = uv[0]
    uv2 = uv[1]
    # mouse_distance 是鼠标到弹弓的距离
    mouse_distance = distance(sr[0], sr[1], sl[0], sl[1])
    # print("distance:", mouse_distance)
    # pu是一个固定的长度
    # rope_lenght = 90
    mouse_distance = 1.0 * mouse_distance / 450 * rope_lenght
    pu = (uv1*rope_lenght+sling_x, uv2*rope_lenght+sling_y)
    put = (uv1*mouse_distance+sling_x, uv2*mouse_distance+sling_y)
    bigger_rope = 102
    x_redbird = put[0] - 20
    y_redbird = put[1] - 20
    if mouse_distance > rope_lenght:
        pux, puy = pu
        pux -= 20
        puy -= 20
        pul = pux, puy
        # screen.blit(redbird, pul)
        pu2 = (uv1*bigger_rope+sling_x, uv2*bigger_rope+sling_y)
        pygame.draw.line(screen, (0, 0, 0), (sling2_x, sling2_y), pu2, 5)
        screen.blit(redbird, pul)
        pygame.draw.line(screen, (0, 0, 0), (sling_x, sling_y), pu2, 5)
    else:
        mouse_distance += 10
        pu3 = (uv1*mouse_distance+sling_x, uv2*mouse_distance+sling_y)
        pygame.draw.line(screen, (0, 0, 0), (sling2_x, sling2_y), pu3, 5)
        screen.blit(redbird, (x_redbird, y_redbird))
        pygame.draw.line(screen, (0, 0, 0), (sling_x, sling_y), pu3, 5)
    # Angle of impulse
    # x是横的，y是竖的
    # 0是横， 1是竖的
    dy = sl[1] - sr[1]
    dx = sl[0] - sr[0]
    if dx == 0:
        dx = 0.00000000000001
    angle = math.atan((float(dy))/dx)
    # print("mouse_distance", mouse_distance)

def draw_level_cleared():
    """Draw level cleared"""
    global game_state
    global bonus_score_once
    global score
    global level_next_time

    

    level_cleared = bold_font3.render("Level Cleared!", 1, WHITE)
    score_level_cleared = bold_font2.render(str(score), 1, WHITE)
    if game_state == 4:
        return
    if level.number_of_birds >= 0 and len(pigs) == 0:
        print("clear!!!!")
        if bonus_score_once:
            score += (level.number_of_birds-1) * 10000
        bonus_score_once = False
        game_state = 4
        rect = pygame.Rect(300, 0, 600, 800)
        pygame.draw.rect(screen, BLACK, rect)
        screen.blit(level_cleared, (450, 90))
        if score >= level.one_star and score <= level.two_star:
            screen.blit(star1, (310, 190))
        if score >= level.two_star and score <= level.three_star:
            screen.blit(star1, (310, 190))
            screen.blit(star2, (500, 170))
        if score >= level.three_star:
            screen.blit(star1, (310, 190))
            screen.blit(star2, (500, 170))
            screen.blit(star3, (700, 200))
        screen.blit(score_level_cleared, (550, 400))
        screen.blit(replay_button, (510, 480))
        screen.blit(next_button, (620, 480))
        level_next_time = time.time()


def draw_level_failed():
    """Draw level failed"""
    global game_state
    global t2
    global birds
    global level_failed_time
    failed = bold_font3.render("Level Failed", 1, WHITE)
    if game_state == 3:
        return
    if level.number_of_birds <= 0 and len(birds) == 0 and time.time() - t2 > 7 and len(pigs) > 0 and game_state != 4:
        game_state = 3
        rect = pygame.Rect(300, 0, 600, 800)
        pygame.draw.rect(screen, BLACK, rect)
        screen.blit(failed, (450, 90))
        screen.blit(pig_happy, (380, 120))
        screen.blit(replay_button, (520, 460))
        level_failed_time = time.time()
        
        


def restart():
    """Delete all objects of the level"""
    pigs_to_remove = []
    birds_to_remove = []
    columns_to_remove = []
    beams_to_remove = []
    for pig in pigs:
        pigs_to_remove.append(pig)
    for pig in pigs_to_remove:
        space.remove(pig.shape, pig.shape.body)
        pigs.remove(pig)
    for bird in birds:
        birds_to_remove.append(bird)
    for bird in birds_to_remove:
        space.remove(bird.shape, bird.shape.body)
        birds.remove(bird)
    for column in columns:
        columns_to_remove.append(column)
    for column in columns_to_remove:
        space.remove(column.shape, column.shape.body)
        columns.remove(column)
    for beam in beams:
        beams_to_remove.append(beam)
    for beam in beams_to_remove:
        space.remove(beam.shape, beam.shape.body)
        beams.remove(beam)


def post_solve_bird_pig(arbiter, space, _):
    """Collision between bird and pig"""
    surface=screen
    a, b = arbiter.shapes
    bird_body = a.body
    pig_body = b.body
    p = to_pygame(bird_body.position)
    p2 = to_pygame(pig_body.position)
    r = 30
    pygame.draw.circle(surface, BLACK, p, r, 4)
    pygame.draw.circle(surface, RED, p2, r, 4)
    pigs_to_remove = []
    for pig in pigs:
        if pig_body == pig.body:
            pig.life -= 20
            pigs_to_remove.append(pig)
            global score
            score += 10000
    for pig in pigs_to_remove:
        space.remove(pig.shape, pig.shape.body)
        pigs.remove(pig)


def post_solve_bird_wood(arbiter, space, _):
    """Collision between bird and wood"""
    poly_to_remove = []
    # fix two impulse length
    if arbiter.total_impulse.length > 1100:
        a, b = arbiter.shapes
        for column in columns:
            if b == column.shape:
                poly_to_remove.append(column)
        for beam in beams:
            if b == beam.shape:
                poly_to_remove.append(beam)
        for poly in poly_to_remove:
            if poly in columns:
                columns.remove(poly)
            if poly in beams:
                beams.remove(poly)
        space.remove(b, b.body)
        global score
        score += 5000


def post_solve_pig_wood(arbiter, space, _):
    """Collision between pig and wood"""
    pigs_to_remove = []
    if arbiter.total_impulse.length > 700:
        pig_shape, wood_shape = arbiter.shapes
        for pig in pigs:
            if pig_shape == pig.shape:
                pig.life -= 20
                global score
                score += 10000
                if pig.life <= 0:
                    pigs_to_remove.append(pig)
    for pig in pigs_to_remove:
        space.remove(pig.shape, pig.shape.body)
        pigs.remove(pig)

class FOClassifer():
    def __init__(self):
        super().__init__()
        self.model = pdx.deploy.Predictor('inference_model', use_gpu=True)
        self.count = 0

    def classifer(self, frame):
        result = self.model.predict(frame.astype('float32'))[0]['category_id']
        if result == 0:
            self.count += 1
            # print("-tforesults", 0)
        else:
            self.count = 0
        # rint("---tforesults count ", self.count)
        if self.count > 2:
            self.count = 0
            return True
        else:
            return False


class MyThread(Thread):
    def __init__(self, video_file = "mykeypointdetector/mine.mp4", debug=True):
        super(MyThread, self).__init__()
        
        self.md = MyDetector("mykeypointdetector/output_inference/hrnet_w32_384x288/", True)
        self.video_file = video_file
        self.debug = debug
        self.capture = cv2.VideoCapture(self.video_file)
        self.h = self.capture.get(4)
        self.w = self.capture.get(3)
        self.FOC = FOClassifer()

    def run(self):
        global results
        global foresults
        global mouse_pressed
        htop = None
        hbottom = None
        hleft = None
        hright = None
        # print("enter detector process", 4)
        radius = 70
        while (1):
            ret, frame = self.capture.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            tresults = self.md.detector.predict(frame)
            
            if tresults['keypoint'][0][0][10][2] > 0.5 and tresults['keypoint'][0][0][8][2] > 0.5:
                
                wx = int(tresults['keypoint'][0][0][10][0])
                wy = int(tresults['keypoint'][0][0][10][1])
                # print(cx , cy)

                ex = int(tresults['keypoint'][0][0][8][0])
                ey = int(tresults['keypoint'][0][0][8][1])

                cx = wx * 2 - ex
                cy = wy * 2 - ey

                cx = int((cx + wx) / 2)
                cy = int((cy + wy) / 2)

                if cx < 0:
                    cx = 0
                if cx >= self.w:
                    cx = self.w - 1
                if cy < 0:
                    cy = 0
                if cy >= self.h:
                    cy = self.h - 1
                
                htop = int(cy - radius) if cy - radius > 0 else 0
                hbottom = int(cy + radius) if cy + radius < self.h else int(self.h-1)
                hleft = int(cx - radius) if cx - radius > 0 else 0
                hright = int(cx + radius) if cx + radius < self.w else int(self.w - 1)
                # print(htop, hbottom, hleft, hright)
                if mouse_pressed:
                    tfo = self.FOC.classifer(frame[htop:hbottom, hleft:hright])
                else:
                    tfo = None
            else:
                tfo = None
            if tresults['keypoint'][0][0][10][2] > 0.5 and tresults['keypoint'][0][0][9][2] > 0.5:
                lock.acquire()
                results = tresults
                foresults = tfo
                lock.release()
            else:
                lock.acquire()
                results = None
                foresults = None
                lock.release()
            '''watch image'''
            if self.debug:
                if tresults is not None:
                    for key in tresults['keypoint'][0][0]:
                        cv2.circle(frame,(int(key[0]), int(key[1])), 5, (0,0,255), -1)
                if tfo is not None:
                    if tfo is True:
                        cv2.rectangle(frame, (hleft, htop), (hright, hbottom), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (hleft, htop), (hright, hbottom), (255, 0, 0), 2)
                cv2.imshow("test", frame)
                cv2.waitKey(1)

# bird and pigs
space.add_collision_handler(0, 1).post_solve=post_solve_bird_pig
# bird and wood
space.add_collision_handler(0, 2).post_solve=post_solve_bird_wood
# pig and wood
space.add_collision_handler(1, 2).post_solve=post_solve_pig_wood
load_music()
level = Level(pigs, columns, beams, space)
level.number = 0
level.load_level()

def main():

    global results
    global foresults
    global static_lines1
    global space
    global level
    
    global mouse_pressed
    global mouse_distance
    global rope_lenght
    global x_mouse
    global y_mouse
    global angle
    global count
    global sling_x
    global sling_y
    global sling2_x
    global sling2_y

    global pigs
    global birds
    global beams
    global columns
    
    global score
    global game_state
    global bird_path
    counter = 0
    restart_counter = False
    global bonus_score_once

    
    global screen
    global redbird
    global background2
    global sling_image
    global pig_image
    global pause_button
    global replay_button
    global play_button
    global next_button
    global clock
    global bold_font
    global t2

    global level_failed_time
    global level_next_time
    
    
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    t1 = 0
    running = True
    wall = False

    pose = None
    shoulderl = None
    shoulderr = None
    wristl = None
    wristr = None
    tforesults = None

    while running:
        if game_state < 3:
            lock.acquire()
            if results is not None:
                if isinstance(results, dict) and 'keypoint' in results.keys():
                    mainres = copy.deepcopy(results['keypoint'])
                    pose = mainres[0][0]
                    shoulderl = (pose[6][0], pose[6][1])
                    shoulderr = (pose[5][0], pose[5][1])
                    wristl = (pose[10][0], pose[10][1])
                    wristr = (pose[9][0], pose[9][1])
                    #elbowl = (pose[8][0], pose[8][1])
            else:
                mainres = None
            tforesults = foresults
            lock.release()
        else:
            pose = None
            tforesults = None
        # print("----tforesults: ", tforesults)

        # tevents= pygame.event.get()
        # print(tevents)
        # print("tevent len", len(tevents))
        for event in pygame.event.get():
            # 退出游戏
            if event.type == pygame.QUIT:
                running = False
            # 退出游戏
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            # 没看懂，按W会触发什么
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                # Toggle wall
                if wall:
                    for line in static_lines1:
                        space.remove(line)
                    wall = False
                else:
                    for line in static_lines1:
                        space.add(line)
                    wall = True
            # 重力缩小
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                space.gravity = (0.0, -10.0)
                level.bool_space = True
            # 重力放大
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                space.gravity = (0.0, -700.0)
                level.bool_space = False
            # 只有在弹弓旁边的电击事件才会被当做是拉动弹弓，mouse_pressed=True
            # 改写的话就是用击掌来判定点击事件
            # 有限状态机:
            # 待机->拉弓->飞行
            '''
            here
            '''
                    
            # 暂停，下一关，重来等
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # 唤出暂停面板
                if (x_mouse < 60 and y_mouse < 155 and y_mouse > 90):
                    game_state = 1
                # 暂停面板出现
                if game_state == 1:
                    # 关闭面板出现
                    if x_mouse > 500 and y_mouse > 200 and y_mouse < 300:
                        # Resume in the paused screen
                        game_state = 0
                    # 重新开始关卡
                    if x_mouse > 500 and y_mouse > 300:
                        # Restart in the paused screen
                        restart()
                        level.load_level()
                        game_state = 0
                        bird_path = []
                # 输掉的时候的面板，重来当局游戏
        if game_state == 3:
            # Restart in the failed level screen
            if time.time() - level_failed_time > 3:
                restart()
                level.load_level()
                game_state = 0
                bird_path = []
                score = 0
                #
        if game_state == 4:
            print("Enter clear!")
            # Build next level
            if time.time() - level_next_time > 3:
                restart()
                level.number += 1
                game_state = 0
                level.load_level()
                score = 0
                bird_path = []
                bonus_score_once = True
        
        # print("mouse_pressed:", mouse_pressed)
        if((pose is not None) and distance(wristl[0], wristl[1], wristr[0], wristr[1]) < 
                    distance(shoulderl[0], shoulderl[1], shoulderr[0], shoulderr[1])):
            # print("enter control")
            mouse_pressed = True
        # 松开鼠标，生成飞出去的小鸟变量
        # 讲胳膊按照脸的比例来进行一定的归一化
        # 然后乘上rope_length
        # angle等于两个手的夹角，即右手是弹弓 左手是拉动的手
        # 抬起鼠标，event.button 为1， 拉弓状态
        # if (event.type == pygame.MOUSEBUTTONUP and event.button == 1 and mouse_pressed):
        if(tforesults is True and mouse_pressed and time.time() * 1000 - t1 > 1000):
            mouse_pressed = False
            if level.number_of_birds > 0:
                level.number_of_birds -= 1
                t1 = time.time() * 1000
                xo = 154
                yo = 156
                mouse_distance = float(mouse_distance)
                if mouse_distance > rope_lenght:
                    mouse_distance = rope_lenght
                # x_mouse 当默认左手总在左边的时候，就会直接使用<的这个选项
                print("angle: ",angle)
                # if x_mouse < sling_x + 5:
                #     bird = Bird(mouse_distance, angle, xo, yo, space)
                #     birds.append(bird)
                # else:
                #     bird = Bird(-mouse_distance, angle, xo, yo, space)
                #     birds.append(bird)
                bird = Bird(mouse_distance, angle, xo, yo, space, time.time())
                birds.append(bird)
                if level.number_of_birds == 0:
                    t2 = time.time()


        # print("enter blend")
        x_mouse, y_mouse = pygame.mouse.get_pos()
        # Draw background
        screen.fill((130, 200, 100))
        screen.blit(background2, (0, -50))
        # Draw first part of the sling
        rect = pygame.Rect(50, 0, 70, 220)
        screen.blit(sling_image, (138, 420), rect)
        # Draw the trail left behind
        for point in bird_path:
            # 23v1 画那个鸟飞的轨迹
            pygame.draw.circle(screen, WHITE, point, 3, 0)
        # Draw the birds in the wait line
        if level.number_of_birds > 0:
            for i in range(level.number_of_birds-1):
                x = 100 - (i*35)
                screen.blit(redbird, (x, 508))
        # Draw sling behavior
        if mouse_pressed and level.number_of_birds > 0:
            # sling_action()
            sling_action_pose(wristl, wristr)
        else:
            if time.time() * 1000 - t1 > 300 and level.number_of_birds > 0:
                screen.blit(redbird, (130, 426))
            else:
                pygame.draw.line(screen, (0, 0, 0), (sling_x, sling_y-8),
                                    (sling2_x, sling2_y-7), 5)
        birds_to_remove = []
        pigs_to_remove = []
        counter += 1
        # Draw birds
        for bird in birds:
            # 23v1 position.y 小于0是啥
            if time.time() - bird.init_time > 10 or bird.shape.body.position.y < 0:
                birds_to_remove.append(bird)
            p = to_pygame(bird.shape.body.position)
            x, y = p
            x -= 22
            y -= 20
            screen.blit(redbird, (x, y))
            # 23v1 这里为什么要画个圆？
            pygame.draw.circle(screen, BLUE, 
                                p, int(bird.shape.radius), 2)
            # 23v1 记录鸟的轨迹
            if counter >= 3 and time.time() - t1 < 5:
                bird_path.append(p)
                restart_counter = True
        if restart_counter:
            counter = 0
            restart_counter = False
        # Remove birds and pigs
        for bird in birds_to_remove:
            space.remove(bird.shape, bird.shape.body)
            birds.remove(bird)
        for pig in pigs_to_remove:
            space.remove(pig.shape, pig.shape.body)
            pigs.remove(pig)
        # Draw static lines
        for line in static_lines:
            body = line.body
            pv1 = body.position + line.a.rotated(body.angle)
            pv2 = body.position + line.b.rotated(body.angle)
            p1 = to_pygame(pv1)
            p2 = to_pygame(pv2)
            pygame.draw.lines(screen, (150, 150, 150), False, [p1, p2])
        i = 0
        # Draw pigs
        for pig in pigs:
            i += 1
            # print(i,pig.life)
            pig = pig.shape
            if pig.body.position.y < 0:
                pigs_to_remove.append(pig)

            p = to_pygame(pig.body.position)
            x, y = p

            angle_degrees = math.degrees(pig.body.angle)
            img = pygame.transform.rotate(pig_image, angle_degrees)
            w, h = img.get_size()
            x -= w * 0.5
            y -= h * 0.5
            screen.blit(img, (x, y))
            pygame.draw.circle(screen, BLUE, p, int(pig.radius), 2)
        # Draw columns and Beams
        for column in columns:
            column.draw_poly('columns', screen)
        for beam in beams:
            beam.draw_poly('beams', screen)
        # Update physics
        dt = 1.0/50.0/2.
        for x in range(2):
            space.step(dt) # make two updates per frame for better stability
        # Drawing second part of the sling
        rect = pygame.Rect(0, 0, 60, 200)
        screen.blit(sling_image, (120, 420), rect)
        # Draw score
        score_font = bold_font.render("SCORE", 1, WHITE)
        number_font = bold_font.render(str(score), 1, WHITE)
        screen.blit(score_font, (1060, 90))
        if score == 0:
            screen.blit(number_font, (1100, 130))
        else:
            screen.blit(number_font, (1060, 130))
        screen.blit(pause_button, (10, 90))
        # Pause option
        if game_state == 1:
            screen.blit(play_button, (500, 200))
            screen.blit(replay_button, (500, 300))
        draw_level_cleared()
        draw_level_failed()
        pygame.display.flip()
        clock.tick(200)
        pygame.display.set_caption("fps: " + str(clock.get_fps()))
        # print("wrist", wristl)
        # print("end blend")


if __name__ == '__main__':
    detectThread = MyThread(0,True)
    detectThread.start()

    main()
    detectThread.join()
    print("Game Shut Down")