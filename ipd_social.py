"""
IPD avec Apprentissage Social - Extension avec imitation et réputation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import copy

# ==================== CONFIGURATION ====================
class SocialConfig:
    # Environnement
    NUM_AGENTS = 6
    NUM_ACTIONS = 2
    STATE_SIZE = 7  # État augmenté avec informations sociales
    
    # Matrice IPD
    REWARD_MATRIX = {
        (0, 0): (3, 3),    # C-C
        (0, 1): (0, 5),    # C-D
        (1, 0): (5, 0),    # D-C
        (1, 1): (1, 1)     # D-D
    }
    
    # DQN
    HIDDEN_SIZE = 64
    LEARNING_RATE = 0.001
    GAMMA = 0.95
    BATCH_SIZE = 32
    MEMORY_SIZE = 5000
    TARGET_UPDATE = 50
    
    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    
    # Entraînement
    NUM_EPISODES = 300
    STEPS_PER_EPISODE = 50
    
    # Mécanismes sociaux
    IMITATION_PROBABILITY = 0.3      # Probabilité d'imiter le leader
    REPUTATION_DECAY = 0.95          # Décroissance de la réputation
    REPUTATION_WEIGHT = 0.5          # Poids de la réputation dans la décision
    
    # Visualisation
    PLOT_WINDOW = 10

config = SocialConfig()

# ==================== ENVIRONNEMENT SOCIAL ====================
class SocialIPDEnvironment:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.agents = list(range(num_agents))
        
        # Historique
        self.last_actions = {i: 0 for i in range(num_agents)}
        self.last_rewards = {i: 3.0 for i in range(num_agents)}
        
        # Métriques sociales
        self.reputations = {i: 1.0 for i in range(num_agents)}
        self.cooperation_counts = {i: 0 for i in range(num_agents)}
        self.total_interactions = {i: 0 for i in range(num_agents)}
        
        # Leader
        self.leader_id = 0
        
        # Statistiques
        self.cooperation_history = []
        self.reward_history = []
        self.reputation_history = []
        self.imitation_history = []
    
    def reset(self):
        self.last_actions = {i: 0 for i in range(self.num_agents)}
        self.last_rewards = {i: 3.0 for i in range(self.num_agents)}
        self.reputations = {i: 1.0 for i in range(self.num_agents)}
        self.cooperation_counts = {i: 0 for i in range(self.num_agents)}
        self.total_interactions = {i: 0 for i in range(self.num_agents)}
        self.leader_id = 0
        
        return self._get_initial_states()
    
    def _get_initial_states(self):
        """États initiaux avec informations sociales"""
        states = {}
        for i in range(self.num_agents):
            partner = random.choice([j for j in range(self.num_agents) if j != i])
            
            states[i] = np.array([
                self.last_actions[i],          # ma dernière action
                self.last_actions[partner],    # dernière action partenaire
                self.last_rewards[i],          # ma dernière récompense
                self.last_rewards[partner],    # dernière récompense partenaire
                self.reputations[i],           # MA réputation
                self.reputations[partner],     # réputation du partenaire
                0                              # action du leader (initial)
            ], dtype=np.float32)
        return states
    
    def update_reputation(self, agent_id, action):
        """Mettre à jour la réputation d'un agent"""
        self.total_interactions[agent_id] += 1
        
        if action == 0:  # Coopération
            self.cooperation_counts[agent_id] += 1
        
        if self.total_interactions[agent_id] > 0:
            coop_rate = self.cooperation_counts[agent_id] / self.total_interactions[agent_id]
            self.reputations[agent_id] = (
                config.REPUTATION_DECAY * self.reputations[agent_id] + 
                (1 - config.REPUTATION_DECAY) * coop_rate
            )
    
    def update_leader(self, rewards):
        """Identifier l'agent leader (meilleure réputation)"""
        self.leader_id = max(self.reputations.items(), key=lambda x: x[1])[0]
        return self.leader_id
    
    def step(self, actions, imitation_flags=None):
        """Exécuter une étape avec tracking social"""
        shuffled = self.agents.copy()
        random.shuffle(shuffled)
        
        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            if i + 1 < len(shuffled):
                pairs.append((shuffled[i], shuffled[i + 1]))
        
        if len(shuffled) % 2 == 1:
            pairs.append((shuffled[-1], shuffled[0]))
        
        rewards = {i: 0.0 for i in range(self.num_agents)}
        
        for a1, a2 in pairs:
            action1 = actions[a1]
            action2 = actions[a2]
            r1, r2 = config.REWARD_MATRIX[(action1, action2)]
            
            rewards[a1] += r1
            rewards[a2] += r2
            
            self.update_reputation(a1, action1)
            self.update_reputation(a2, action2)
        
        # Mettre à jour le leader
        leader_id = self.update_leader(rewards)
        leader_action = actions.get(leader_id, 0)
        
        # Statistiques
        coop_count = sum(1 for a in actions.values() if a == 0)
        coop_rate = coop_count / len(actions) if len(actions) > 0 else 0
        
        self.cooperation_history.append(coop_rate)
        self.reward_history.append(np.mean(list(rewards.values())))
        self.reputation_history.append(np.mean(list(self.reputations.values())))
        
        imitation_rate = 0
        if imitation_flags:
            imitation_rate = sum(imitation_flags.values()) / len(imitation_flags) if imitation_flags else 0
        self.imitation_history.append(imitation_rate)
        
        # Nouveaux états avec informations sociales
        next_states = {}
        for i in range(self.num_agents):
            possible_partners = [j for j in range(self.num_agents) if j != i]
            partner = random.choice(possible_partners) if possible_partners else i
            
            next_states[i] = np.array([
                actions[i],
                actions[partner] if partner != i else 0,
                rewards[i],
                rewards[partner] if partner != i else rewards[i],
                self.reputations[i],
                self.reputations[partner] if partner != i else self.reputations[i],
                leader_action
            ], dtype=np.float32)
        
        self.last_actions = actions.copy()
        self.last_rewards = rewards.copy()
        
        return next_states, rewards, leader_id

# ==================== AGENT AVEC APPRENTISSAGE SOCIAL ====================
class SocialDQNAgent:
    def __init__(self, agent_id, state_size, action_size, hidden_size):
        self.id = agent_id
        
        # Réseau DQN
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        # Exploration
        self.epsilon = config.EPSILON_START
        self.steps_done = 0
        
        # Métriques sociales
        self.imitation_count = 0
    
    def select_action(self, state, social_info=None):
        """Sélectionner une action avec mécanismes sociaux"""
        self.steps_done += 1
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)
        
        # 1. Mécanisme d'imitation
        if social_info and 'leader_action' in social_info:
            if random.random() < config.IMITATION_PROBABILITY:
                self.imitation_count += 1
                return social_info['leader_action']
        
        # 2. Exploration epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, config.NUM_ACTIONS - 1)
        
        # 3. Décision basée sur le réseau avec ajustement social
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            
            # Ajustement basé sur la réputation du partenaire
            if social_info and 'partner_reputation' in social_info:
                partner_rep = social_info['partner_reputation']
                
                adjusted_q = q_values.clone()
                
                # Favoriser la coopération avec partenaires fiables
                if partner_rep > 0.5:
                    adjusted_q[0, 0] += config.REPUTATION_WEIGHT * partner_rep
                else:
                    adjusted_q[0, 1] += config.REPUTATION_WEIGHT * (1 - partner_rep)
                
                return adjusted_q.argmax().item()
            else:
                return q_values.argmax().item()
    
    def optimize(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, config.BATCH_SIZE)
        
        states_array = np.array([exp[0] for exp in batch], dtype=np.float32)
        next_states_array = np.array([exp[3] for exp in batch], dtype=np.float32)
        
        states = torch.FloatTensor(states_array)
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor(next_states_array)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + config.GAMMA * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

# ==================== ENTRAÎNEMENT ====================
def train_social_model():
    print("=" * 60)
    print("IPD DQN - AVEC APPRENTISSAGE SOCIAL")
    print("=" * 60)
    print("Mécanismes sociaux activés:")
    print(f"  • Imitation (probabilité: {config.IMITATION_PROBABILITY})")
    print(f"  • Réputation (poids: {config.REPUTATION_WEIGHT})")
    print("=" * 60)
    
    start_time = time.time()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    env = SocialIPDEnvironment(config.NUM_AGENTS)
    agents = [SocialDQNAgent(i, config.STATE_SIZE, config.NUM_ACTIONS, config.HIDDEN_SIZE) 
              for i in range(config.NUM_AGENTS)]
    
    stats = {
        'cooperation': [],
        'rewards': [],
        'epsilon': [],
        'reputation': [],
        'imitation': [],
        'memory_sizes': []
    }
    
    for episode in range(config.NUM_EPISODES):
        states = env.reset()
        episode_coop = []
        episode_rewards = []
        episode_imitation = 0
        
        for step in range(config.STEPS_PER_EPISODE):
            leader_id = env.leader_id
            leader_action = env.last_actions.get(leader_id, 0)
            
            imitation_flags = {}
            actions = {}
            
            for i, agent in enumerate(agents):
                # Information sociale
                other_reputations = [env.reputations[j] for j in range(config.NUM_AGENTS) if j != i]
                avg_partner_rep = np.mean(other_reputations) if other_reputations else 0.5
                
                social_info = {
                    'leader_action': leader_action,
                    'partner_reputation': avg_partner_rep
                }
                
                action = agent.select_action(states[i], social_info)
                actions[i] = action
                imitation_flags[i] = (action == leader_action)
            
            next_states, rewards, leader_id = env.step(actions, imitation_flags)
            
            for i, agent in enumerate(agents):
                agent.save_experience(
                    states[i],
                    actions[i],
                    float(rewards[i]),
                    next_states[i]
                )
            
            if step % 2 == 0:
                for agent in agents:
                    agent.optimize()
            
            if step % config.TARGET_UPDATE == 0:
                for agent in agents:
                    agent.update_target()
            
            states = next_states
            episode_coop.append(sum(1 for a in actions.values() if a == 0) / len(actions))
            episode_rewards.append(np.mean(list(rewards.values())))
            episode_imitation += sum(imitation_flags.values()) / len(actions)
        
        stats['cooperation'].append(np.mean(episode_coop))
        stats['rewards'].append(np.mean(episode_rewards))
        stats['epsilon'].append(np.mean([a.epsilon for a in agents]))
        stats['reputation'].append(np.mean(list(env.reputations.values())))
        stats['imitation'].append(episode_imitation / config.STEPS_PER_EPISODE)
        stats['memory_sizes'].append(np.mean([len(a.memory) for a in agents]))
        
        if (episode + 1) % 30 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
            remaining = (config.NUM_EPISODES - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
            
            print(f"Épisode {episode + 1:3d}/{config.NUM_EPISODES} | "
                  f"Coop: {stats['cooperation'][-1]:.3f} | "
                  f"Reward: {stats['rewards'][-1]:.3f} | "
                  f"Rep: {stats['reputation'][-1]:.3f} | "
                  f"Imit: {stats['imitation'][-1]:.3f} | "
                  f"Temps: {elapsed:.0f}s (+{remaining:.0f}s)")
    
    total_time = time.time() - start_time
    print(f"\n✅ Entraînement terminé en {total_time:.1f} secondes")
    return env, agents, stats, total_time

# ==================== VISUALISATION ====================
def plot_social_results(stats, total_time):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    axes[0, 0].plot(stats['cooperation'], alpha=0.7, color='blue')
    axes[0, 0].set_title('Taux de Coopération')
    axes[0, 0].set_xlabel('Épisode')
    axes[0, 0].set_ylabel('Taux')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(stats['rewards'], alpha=0.7, color='orange')
    axes[0, 1].set_title('Récompense Moyenne')
    axes[0, 1].set_xlabel('Épisode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(stats['reputation'], alpha=0.7, color='green')
    axes[1, 0].set_title('Réputation Moyenne')
    axes[1, 0].set_xlabel('Épisode')
    axes[1, 0].set_ylabel('Réputation')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(stats['imitation'], alpha=0.7, color='purple')
    axes[1, 1].set_title('Taux d\'Imitation')
    axes[1, 1].set_xlabel('Épisode')
    axes[1, 1].set_ylabel('Imitation rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].plot(stats['epsilon'], alpha=0.7, color='red')
    axes[2, 0].set_title('Epsilon (Exploration)')
    axes[2, 0].set_xlabel('Épisode')
    axes[2, 0].set_ylabel('Epsilon')
    axes[2, 0].grid(True, alpha=0.3)
    
    window = config.PLOT_WINDOW
    smoothed_coop = pd.Series(stats['cooperation']).rolling(
        window=window, min_periods=1
    ).mean()
    
    axes[2, 1].plot(stats['cooperation'], alpha=0.3, label='Brut', color='blue')
    axes[2, 1].plot(smoothed_coop, alpha=0.9, linewidth=2, 
                   label=f'Moyenne {window} épisodes', color='darkblue')
    axes[2, 1].set_title('Taux de Coopération (lissé)')
    axes[2, 1].set_xlabel('Épisode')
    axes[2, 1].set_ylabel('Taux')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'IPD avec Apprentissage Social - {config.NUM_AGENTS} agents - {total_time:.0f}s', 
                fontsize=14)
    plt.tight_layout()
    plt.savefig('ipd_social_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return smoothed_coop

# ==================== ANALYSE ====================
def analyze_social_strategies(env, agents, stats):
    print("\n" + "=" * 60)
    print("ANALYSE DES STRATÉGIES SOCIALES")
    print("=" * 60)
    
    final_coop = np.mean(stats['cooperation'][-50:]) if len(stats['cooperation']) >= 50 else np.mean(stats['cooperation'])
    final_reward = np.mean(stats['rewards'][-50:]) if len(stats['rewards']) >= 50 else np.mean(stats['rewards'])
    
    print(f"\n📊 RÉSULTATS FINAUX:")
    print(f"  • Taux de coopération: {final_coop:.3f}")
    print(f"  • Récompense moyenne: {final_reward:.3f}")
    print(f"  • Réputation moyenne: {np.mean(stats['reputation'][-10:]):.3f}")
    print(f"  • Taux d'imitation: {np.mean(stats['imitation'][-10:]):.3f}")
    
    baseline_coop = 0.005
    baseline_reward = 1.016
    
    print(f"\n📈 COMPARAISON AVEC BASELINE:")
    print(f"  Baseline:  coopération={baseline_coop:.3f}, récompense={baseline_reward:.3f}")
    print(f"  Social:    coopération={final_coop:.3f}, récompense={final_reward:.3f}")
    
    if final_coop > baseline_coop:
        improvement = ((final_coop - baseline_coop) / baseline_coop * 100)
        print(f"  Amélioration: +{improvement:.1f}%")
    
    print(f"\n🔍 COMPORTEMENT DES AGENTS:")
    for i, agent in enumerate(agents[:2]):
        print(f"\nAgent {i}:")
        print(f"  • Imitations: {agent.imitation_count}")
        print(f"  • Expériences: {len(agent.memory)}")
        print(f"  • Stratégie: ", end="")
        
        test_state = np.array([0, 0, 3, 3, 0.8, 0.8, 0], dtype=np.float32)
        with torch.no_grad():
            q_vals = agent.policy_net(torch.FloatTensor(test_state).unsqueeze(0))
            if q_vals[0][0] > q_vals[0][1]: 
                print("Coopère avec partenaires fiables")
            else:
                print("Défaut même avec partenaires fiables")

# ==================== EXÉCUTION ====================
if __name__ == "__main__":
    print("=" * 60)
    print("PROJET IPD - APPRENTISSAGE SOCIAL")
    print("=" * 60)
    print("Extension avec:")
    print("  1. Imitation du leader (30% de probabilité)")
    print("  2. Réputation des partenaires (influence les décisions)")
    print("  3. État augmenté (7 dimensions)")
    print("=" * 60)
    
    # Entraînement
    env, agents, stats, total_time = train_social_model()
    
    # Visualisation
    smoothed_coop = plot_social_results(stats, total_time)
    
    # Analyse
    analyze_social_strategies(env, agents, stats)
    
    # Sauvegarde
    summary = {
        "social_learning_results": {
            "total_training_time": total_time,
            "final_cooperation_rate": float(np.mean(stats['cooperation'][-50:]) if len(stats['cooperation']) >= 50 else np.mean(stats['cooperation'])),
            "final_average_reward": float(np.mean(stats['rewards'][-50:]) if len(stats['rewards']) >= 50 else np.mean(stats['rewards'])),
            "average_reputation": float(np.mean(stats['reputation'][-50:]) if len(stats['reputation']) >= 50 else np.mean(stats['reputation'])),
            "total_imitations": sum([a.imitation_count for a in agents]),
            "comparison_with_baseline": {
                "baseline_cooperation": 0.005,
                "baseline_reward": 1.016,
                "improvement_cooperation_percent": float(((np.mean(stats['cooperation'][-50:] if len(stats['cooperation']) >= 50 else np.mean(stats['cooperation'])) - 0.005) / 0.005 * 100) if 0.005 > 0 else 0)
            }
        }
    }
    
    with open('ipd_social_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Résultats sauvegardés dans 'ipd_social_summary.json'")
    print(f"✅ Graphiques sauvegardés dans 'ipd_social_results.png'")
    
    print("\n" + "=" * 60)
    print("ANALYSE TERMINÉE")
    print("=" * 60)
    print("L'apprentissage social montre:")
    print("  • Impact des mécanismes d'imitation et réputation")
    print("  • Évolution de la coopération dans la population")
    print("  • Comparaison quantitative avec la baseline")
    print("=" * 60)