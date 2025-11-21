import pygame
import sys
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize pygame
pygame.init()

# Game window size
WIDTH, HEIGHT = 600, 400
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Pong")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle and ball settings
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 60
PADDLE_SPEED = 5
BALL_SIZE = 10
BALL_MIN_SPEED = 3
BALL_MAX_SPEED = 4

MAX_SCORE = 5

clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 36)

class GameEnv:

    def __init__(self, agent):
        self.player_score = 0
        self.enemy_score = 0
        self.last_scored = 0  # 0 - no one yet, 1 - player, 2 - enemy
        
        self.agent = agent

        self.reset()

    # Starts a completely new game
    def new_game(self):
        self.player_score = 0
        self.enemy_score = 0
        self.last_scored = 0
        self.reset()

    # Resets the current state to the initial
    def reset(self):
        # Positions
        self.player_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.enemy_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.ball_x, self.ball_y = WIDTH // 2, HEIGHT // 2

        # If enemy scored, let player start
        if self.last_scored == 2:
            self.ball_vx = random.uniform(-BALL_MIN_SPEED, -BALL_MAX_SPEED)
            self.ball_vy = random.choice([random.uniform(-BALL_MAX_SPEED, -BALL_MIN_SPEED), random.uniform(BALL_MIN_SPEED, BALL_MAX_SPEED)])
        # Otherwise, let enemy start
        else:
            self.ball_vx = random.uniform(BALL_MIN_SPEED, BALL_MAX_SPEED)
            self.ball_vy = random.choice([random.uniform(-BALL_MAX_SPEED, -BALL_MIN_SPEED), random.uniform(BALL_MIN_SPEED, BALL_MAX_SPEED)])

    # state = [ball_x, ball_y, ball_vx, ball_vy, player_y]
    def extract_game_state(self):
        normalized_ball_x = self.ball_x / WIDTH
        normalized_ball_y = self.ball_y / HEIGHT
        normalized_ball_vx = self.ball_vx / BALL_MAX_SPEED
        normalized_ball_vy = self.ball_vy / BALL_MAX_SPEED
        normalized_enemy_y = self.enemy_y / HEIGHT
        state = torch.tensor([
            normalized_ball_x,
            normalized_ball_y,
            normalized_ball_vx,
            normalized_ball_vy,
            normalized_enemy_y
        ], dtype=torch.float32)

        return state.to(self.agent.device)
    
    
    def render(self):
        win.fill(BLACK)

        # Draw paddles and ball, according to the position
        pygame.draw.rect(win, WHITE, (20, self.player_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(win, WHITE, (WIDTH - 30, self.enemy_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(win, WHITE, (self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE))
        # Text to show both players' score
        score_text = font.render(f"{self.player_score} : {self.enemy_score}", True, WHITE)
        win.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

        # Display update
        pygame.display.update()
        clock.tick(60)


    def step(self, action):
        reward = 0

        # Still = 0
        # Up = 1
        if(action == 1):
            self.enemy_y -= PADDLE_SPEED
        # Down = 2
        elif(action == 2):
            self.enemy_y += PADDLE_SPEED

        # Adding collisions
        player_rect = pygame.Rect(20, self.player_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        enemy_rect = pygame.Rect(WIDTH - 30, self.enemy_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE)

        if ball_rect.colliderect(player_rect):
            # Move the ball to the other direction when it hits the paddle
            self.ball_vx *= -1
            self.ball_x = player_rect.right
        if ball_rect.colliderect(enemy_rect):
            # Positive reward for hitting the ball
            reward += 1.0
            # Move the ball to the other direction when it hits the paddle
            self.ball_vx *= -1
            self.ball_x = enemy_rect.left - BALL_SIZE 
        
        # Keep enemy in bounds
        self.enemy_y = max(0, min(HEIGHT - PADDLE_HEIGHT, self.enemy_y))

        # Update ball position
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        # Bounce off top/bottom
        if self.ball_y <= 0 or self.ball_y >= HEIGHT - BALL_SIZE:
            self.ball_vy *= -1

        next_state = self.extract_game_state()

        # Point for enemy (AI)
        if self.ball_x <= 0:
            reward += 10
            done = self.scored(2)
            self.reset()
            return next_state, reward, done
        # Point for player (input or dummy)
        elif self.ball_x >= WIDTH:
            reward -= 10
            done = self.scored(1)
            self.reset()
            return next_state, reward, done
        else:
            # Gives reward based on the distance in Y axis from the ball
            distance = abs((self.enemy_y + PADDLE_HEIGHT // 2) - self.ball_y)
            proximity = 1 - (distance / HEIGHT)  # Normalize to [0, 1]
            reward += proximity * 0.1
            return next_state, reward, False


    # Game loop
    def run_episode_with_display(self):
        while self.game_finished() == 0:
            # Handle quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Player movement
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.player_y -= PADDLE_SPEED
            if keys[pygame.K_s]:
                self.player_y += PADDLE_SPEED

            self.render()

            state = self.extract_game_state()
            self.step(agent.act(state)) # AI movement

            # Keep players in bounds
            self.player_y = max(0, min(HEIGHT - PADDLE_HEIGHT, self.player_y))

        # End screen: showing won or losed
        win.fill(BLACK)
        if env.game_finished() == 1:
            message = "You Win!"
        else:
            message = "You Lose!"

        text = font.render(message, True, WHITE)
        win.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 20))
        pygame.display.update()
        pygame.time.delay(2000)

    # Returns whether someone scored or not
    def scored(self, player):
        self.last_scored = player
        if(player == 1):
            self.player_score += 1
        else:
            self.enemy_score += 1

        return self.game_finished() != 0
        
    # 0 = not finished, 1 = player won, 2 = enemy won
    def game_finished(self):
        if MAX_SCORE <= self.player_score:
            return 1
        elif MAX_SCORE <= self.enemy_score:
            return 2
        else:
            return 0


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 128)  # 5 inputs → 128 hidden
        self.fc2 = nn.Linear(128, 128) # hidden → hidden
        self.fc3 = nn.Linear(128, 3)  # hidden → 3 actions

    def forward(self, x):
        x = F.relu(self.fc1(x))     # Apply ReLU to layer 1
        x = F.relu(self.fc2(x))     # Apply ReLU to layer 2
        x = self.fc3(x)             # Final output layer (raw Q-values)
        return x

class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.memory = deque(maxlen=50000)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.learning_rate = 0.00025
        self.batch_size = 128
        self.update_target_every = 100
        self.episode_count = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Move model and input to same device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)

    # Adds a new memory
    def remember(self, state, action, reward, next_state, done):
        # state and next_state are stored in CPU tensors to reduce GPU memory usage
        self.memory.append((state.cpu(), action, reward, next_state.cpu(), done))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Chooses the next step
    def act(self, state):
        state = state.to(self.device) # Ensure state is on the correct device
        
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            with torch.no_grad():
                return torch.argmax(self.model.forward(state)).item()

    
    # Trains the model
    def replay(self):
        # Randomly choosing a batch of memories inside the memory deque
        batch = random.sample(self.memory, self.batch_size)

        for memory in batch:
            state, action, reward, next_state, done = memory

            state = state.to(self.device)
            next_state = next_state.to(self.device)

            if(done):
                target = reward
            else:
                next_q = torch.max(self.target_model.forward(next_state)).item()
                target = reward + self.gamma * next_q

            # Predicts Q-values for all actions in the current state (using the model)
            predicted_qs = self.model.forward(state)
            # Copies the predicted Q-values and replaces the Q-value of the action taken
            target_qs = predicted_qs.clone().detach()
            target_qs[action] = target

            # Calculates the difference between the versions of the neuron network
            loss = F.mse_loss(predicted_qs, target_qs)
            # Improves model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  # Updates the model

            self.episode_count += 1
            if self.episode_count % self.update_target_every == 0:
                self.target_model.load_state_dict(self.model.state_dict())



episodes = 501
display_episodes = [0, 50, 100, 200, 300, 500]
max_steps_per_episode = 3000

agent = DQNAgent()
envs = [GameEnv(agent) for game in range(8)] # Run 8 games in parallel


for episode in range(episodes):
    
    print("Running episode:", episode)

    # Just display, don't train in this time
    if(episode in display_episodes):
        input()
        for i in range(4):
            env = envs[i] # Only one game environment for each when displaying
            env.new_game()
            # Plays the game with player input
            env.run_episode_with_display()

    # Playing in parallel multiple Pongs
    else:
        envs_finished = [False for env in range(len(envs))]

        for env in envs:
            env.new_game()  # Resets board before start

        for step in range(max_steps_per_episode):
            for i in range(len(envs)):
                if(envs_finished[i] == False):
                    env = envs[i]

                    state = env.extract_game_state()

                    # Simple player dummy AI against the agent
                    if env.player_y + PADDLE_HEIGHT // 2 < env.ball_y:
                        env.player_y += PADDLE_SPEED
                    else:
                        env.player_y -= PADDLE_SPEED
                    # Keep players in bounds
                    env.player_y = max(0, min(HEIGHT - PADDLE_HEIGHT, env.player_y))

                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.remember(state, action, reward, next_state, done)

                    # If game has ended, start a new one
                    if done:
                        envs_finished[i] = True

            # Every few steps, train the agent when enough memory was collected
            if step % 20 == 0 and len(agent.memory) > agent.batch_size:
                # When the memory is large, train more batches at a time
                if len(agent.memory) > agent.batch_size * 2:
                    for i in range(5):
                        agent.replay()

                agent.replay()


    agent.decay_epsilon() # Decaying epsilon after every episode
    
    for i, env in enumerate(envs):
        winner = env.game_finished()
        if winner:
            outcome = "Player Won" if winner == 1 else "Enemy Won"
        else:
            outcome = "Incomplete"
        print(f"Env {i} | Episode {episode} | Score: {env.player_score}:{env.enemy_score} | {outcome}")
