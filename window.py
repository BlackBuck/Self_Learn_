import pygame
from sys import exit
from ann import ann
import scipy
import random

pygame.init()
screen = pygame.display.set_mode((800, 400))
pygame.display.set_caption("Self Player")
clock = pygame.time.Clock()

font = pygame.font.Font(None, 50)

#player score and lives
score = 0
lives = 300000
score_txt = "Score : " + str(score)
life_txt = "Lives : " + str(lives)

testSurface2 = pygame.image.load('sky.jpg')
score_surf = font.render(score_txt, False, 'brown')
score_rect = score_surf.get_rect(topleft=(10, 10))
life_surf = font.render(life_txt, False, 'brown')
life_rect = life_surf.get_rect(topleft=(10, 45))

#snail
snail = pygame.image.load('snail1.png').convert_alpha()
snail_rect = snail.get_rect(midbottom = (800, 300))
obstacles = [snail_rect]

#ground and sky
ground = pygame.image.load('ground.png').convert()
ground_rect = ground.get_rect(topleft=(0,300))
sky = pygame.image.load('Sky.png').convert()

#player
player_surf = pygame.image.load('player_walk_1.png').convert_alpha()
player_rect = player_surf.get_rect(midbottom=(200, 299))

snail_pos_x = 600

gravity = 0

#checks if game is paused or not
paused = True

#activation function for the neural network
activation_function = lambda x: scipy.special.expit(x)

#the neural network
nn = ann(4, 5, 1, 0.3)
output = [[0]]
req_output = [1]
prev_lives = lives
input = []

#obstacle timing
obstacle_timer = pygame.USEREVENT + 1
pygame.time.set_timer(obstacle_timer, 900)

#keep a track of past few lives' scores
scores = []

while True:
    #for the autonomous part
    prev_lives = lives
    #update the score and life text
    score_txt = "Score : " + str(score)
    life_txt = "Lives : " + str(lives)
    score_surf = font.render(score_txt, False, "brown")
    life_surf = font.render(life_txt, False, "brown")
    
    #player collision with the snail
    if player_rect.colliderect(obstacles[0]):
        lives -= 1
        scores.append(score)
        if output[0] > 0.6:
            req_output = [1-output[0][0]]
        else:
            req_output = [1-output[0][0]]
        nn.train(input, req_output)
        # print(req_output)
        obstacles.pop(0)
    if lives == 0:
        # pygame.time.delay(30000)
        paused = True

    if paused:
        pygame.display.flip()
        screen.blit(sky, (0,0))
        screen.blit(ground, ground_rect)
        screen.blit(score_surf, (350, 10))
        screen.blit(life_surf, (350, 50))
        
        pygame.draw.rect(screen, "black", pygame.Rect(320, 100, 220, 200), 0, 5)
        screen.blit(font.render("Paused", False, "white"), (350, 110))

        #the restart option
        restart = font.render("New Game", False, "white")
        res_rect = restart.get_rect(topleft=(350, 150))
        pygame.draw.rect(screen, "brown", res_rect)
        screen.blit(restart, (350, 150))
        if res_rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(screen, "white", res_rect, 2)
        
        #the quit option
        quit = font.render("Quit", False, "white")
        quit_rect = quit.get_rect(topleft=(350, 250))
        pygame.draw.rect(screen, "brown", quit_rect)
        screen.blit(quit, (350, 250))
        if quit_rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(screen, "white",quit_rect, 2)

        #the continue option
        resume = font.render("Resume", False, "white")
        resume_rect = resume.get_rect(topleft=(350, 200))
        pygame.draw.rect(screen, "brown", resume_rect)
        screen.blit(resume, (350, 200))
        if resume_rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(screen, "white",resume_rect, 2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if quit_rect.collidepoint(pygame.mouse.get_pos()):
                    pygame.quit()
                    exit()
                if resume_rect.collidepoint(pygame.mouse.get_pos()):
                    paused = False
                if res_rect.collidepoint(pygame.mouse.get_pos()):
                    score = 0
                    lives = 3000
                    paused = False
            

    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and player_rect.bottom > 280:
                    gravity = -20
                    
                if event.key == pygame.K_p:
                    paused = True
            if event.type == obstacle_timer:
                obstacles.append(snail.get_rect(midbottom = (random.randint(obstacles[len(obstacles)-1].right+400, obstacles[len(obstacles)-1].right+600), 300)))
                

        #block image transfer
        #link the test surface to the screen
        screen.blit(sky, (0, 0))
        
        #drawing the snails
        for enemy in obstacles:
            screen.blit(snail, enemy)

        screen.blit(ground, (0, 300))
        # pygame.draw.rect(screen, "pink", score_rect)
        # pygame.draw.rect(screen, "white", score_rect, 2)
        screen.blit(score_surf, score_rect)
        # pygame.draw.rect(screen, "pink", life_rect)
        # pygame.draw.rect(screen, "white", life_rect, 2)
        screen.blit(life_surf, life_rect)

        #player
        gravity += 1
        player_rect.y += gravity
        if player_rect.bottom > 302: player_rect.bottom = 300
        if player_rect.top < 2: player_rect.top = 0
        screen.blit(player_surf, player_rect)
        
        #updating the snails
        for enemy in obstacles:
            enemy.right -= 5
            if(enemy.right < -50):
                obstacles.pop(0)
                score += 1
                nn.train(input, output)
        
        #the autonomous part
        input = [obstacles[0].width, obstacles[0].height, 5, obstacles[0].left - player_rect.right]
        # input = [obstacles[0].width, obstacles[0].height, 5, obstacles[0].left - player_rect.right, obstacles[1].left - player_rect.right]
        output = nn.query(input)
        print(output)
        if(output[0][0] > 0.65 and player_rect.bottom > 295):
            gravity = -20
            if player_rect.colliderect(obstacles[0]):
                print("Changing")
                req_output = [1-output[0][0]]
                nn.train(input, req_output)
        else:
            if player_rect.colliderect(obstacles[0]):
                req_output = [1-output[0][0]]
                nn.train(input, req_output)
                pass

        #update the display
        # nn.train(req_output)
        slen = len(scores)
        if lives%7 == 0 and scores[slen - 1] == scores[slen - 2] and scores[slen - 2] == scores[slen - 3] and scores[slen - 1] < 30:
            nn = ann(4, 5, 1, 0.3)
            scores = [1, 2, 3]
        pygame.display.flip()
        clock.tick(60)

    

