"""
IPD Baseline - Version COMPLÈTE SANS FICHIERS .pth
Avec affichage terminal et sauvegarde JSON
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import pandas as pd
import time
import json

# ==================== CONFIGURATION ====================
class Config:
    # Environnement
    NUM_AGENTS = 6           # Réduit pour aller plus vite
    NUM_ACTIONS = 2
    STATE_SIZE = 4
    
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
    
    # Visualisation
    PLOT_WINDOW = 10

config = Config()

# ==================== ENVIRONNEMENT ====================
class IPDEnvironment:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.agents = list(range(num_agents))
        
        self.last_actions = {i: 0 for i in range(num_agents)}
        self.last_rewards = {i: 3.0 for i in range(num_agents)}
        
        self.cooperation_history = []
        self.reward_history = []
    
    def reset(self):
        self.last_actions = {i: 0 for i in range(self.num_agents)}
        self.last_rewards = {i: 3.0 for i in range(self.num_agents)}
        return self._get_initial_states()
    
    def _get_initial_states(self):
        states = {}
        for i in range(self.num_agents):
            partner = random.choice([j for j in range(self.num_agents) if j != i])
            states[i] = np.array([
                self.last_actions[i],
                self.last_actions[partner],
                self.last_rewards[i],
                self.last_rewards[partner]
            ], dtype=np.float32)
        return states
    
    def step(self, actions):
        shuffled = self.agents.copy()
        random.shuffle(shuffled)
        
        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            if i + 1 < len(shuffled):
                pairs.append((shuffled[i], shuffled[i + 1]))
        
        if len(shuffled) % 2 == 1:
            pairs.append((shuffled[-1], shuffled[0]))
        
        rewards = {i: 0.0 for i in range(self.num_agents)}
        next_states = {}
        
        for a1, a2 in pairs:
            action1 = actions[a1]
            action2 = actions[a2]
            r1, r2 = config.REWARD_MATRIX[(action1, action2)]
            rewards[a1] += r1
            rewards[a2] += r2
        
        coop_count = sum(1 for a in actions.values() if a == 0)
        self.cooperation_history.append(coop_count / len(actions))
        self.reward_history.append(np.mean(list(rewards.values())))
        
        for i in range(self.num_agents):
            possible_partners = [j for j in range(self.num_agents) if j != i]
            partner = random.choice(possible_partners) if possible_partners else i
            
            next_states[i] = np.array([
                actions[i],
                actions[partner] if partner != i else 0,
                rewards[i],
                rewards[partner] if partner != i else rewards[i]
            ], dtype=np.float32)
        
        self.last_actions = actions.copy()
        self.last_rewards = rewards.copy()
        
        return next_states, rewards, None

# ==================== RÉSEAU DQN ====================
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# ==================== MÉMOIRE ====================
Transition = namedtuple('Transition', 
                       ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# ==================== AGENT DQN ====================
class DQNAgent:
    def __init__(self, agent_id, state_size, action_size, hidden_size):
        self.id = agent_id
        
        self.policy_net = DQN(state_size, hidden_size, action_size)
        self.target_net = DQN(state_size, hidden_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = ReplayMemory(config.MEMORY_SIZE)
        
        self.epsilon = config.EPSILON_START
        self.steps_done = 0
    
    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)
        
        if random.random() < self.epsilon:
            return random.randint(0, config.NUM_ACTIONS - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def optimize(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(batch.next_state)
        done_batch = torch.FloatTensor(batch.done)
        
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + config.GAMMA * next_q * (1 - done_batch)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_experience(self, state, action, reward, next_state, done=0.0):
        self.memory.push(state, action, reward, next_state, done)

# ==================== ENTRAÎNEMENT ====================
def train_model():
    print("=" * 60)
    print("IPD DQN - ENTRAÎNEMENT")
    print("=" * 60)
    
    start_time = time.time()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    env = IPDEnvironment(config.NUM_AGENTS)
    agents = [DQNAgent(i, config.STATE_SIZE, config.NUM_ACTIONS, config.HIDDEN_SIZE) 
              for i in range(config.NUM_AGENTS)]
    
    stats = {
        'cooperation': [],
        'rewards': [],
        'epsilon': [],
        'memory_sizes': []
    }
    
    for episode in range(config.NUM_EPISODES):
        states = env.reset()
        episode_coop = []
        episode_rewards = []
        
        for step in range(config.STEPS_PER_EPISODE):
            actions = {}
            for i, agent in enumerate(agents):
                actions[i] = agent.select_action(states[i])
            
            next_states, rewards, _ = env.step(actions)
            
            for i, agent in enumerate(agents):
                agent.save_experience(
                    states[i],
                    actions[i],
                    float(rewards[i]),
                    next_states[i],
                    0.0
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
        
        stats['cooperation'].append(np.mean(episode_coop))
        stats['rewards'].append(np.mean(episode_rewards))
        stats['epsilon'].append(np.mean([a.epsilon for a in agents]))
        stats['memory_sizes'].append(np.mean([len(a.memory) for a in agents]))
        
        if (episode + 1) % 30 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
            remaining = (config.NUM_EPISODES - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
            
            print(f"Épisode {episode + 1:3d}/{config.NUM_EPISODES} | "
                  f"Coop: {stats['cooperation'][-1]:.3f} | "
                  f"Reward: {stats['rewards'][-1]:.3f} | "
                  f"Eps: {stats['epsilon'][-1]:.3f} | "
                  f"Temps: {elapsed:.0f}s (+{remaining:.0f}s)")
    
    total_time = time.time() - start_time
    print(f"\n✅ Entraînement terminé en {total_time:.1f} secondes")
    return env, agents, stats, total_time

# ==================== VISUALISATION ====================
def plot_results(stats, total_time):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(stats['cooperation'], alpha=0.7)
    axes[0, 0].set_title('Taux de Coopération')
    axes[0, 0].set_xlabel('Épisode')
    axes[0, 0].set_ylabel('Taux')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(stats['rewards'], alpha=0.7, color='orange')
    axes[0, 1].set_title('Récompense Moyenne')
    axes[0, 1].set_xlabel('Épisode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(stats['epsilon'], alpha=0.7, color='green')
    axes[1, 0].set_title('Epsilon (Exploration)')
    axes[1, 0].set_xlabel('Épisode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True, alpha=0.3)
    
    window = config.PLOT_WINDOW
    smoothed_coop = pd.Series(stats['cooperation']).rolling(
        window=window, min_periods=1
    ).mean()
    
    axes[1, 1].plot(stats['cooperation'], alpha=0.3, label='Brut')
    axes[1, 1].plot(smoothed_coop, alpha=0.9, linewidth=2, 
                   label=f'Moyenne {window} épisodes')
    axes[1, 1].set_title('Taux de Coopération (lissé)')
    axes[1, 1].set_xlabel('Épisode')
    axes[1, 1].set_ylabel('Taux')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'IPD avec DQN - {config.NUM_AGENTS} agents - {total_time:.0f}s', 
                fontsize=14)
    plt.tight_layout()
    plt.savefig('ipd_complete_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return smoothed_coop

# ==================== ANALYSE DES STRATÉGIES ====================
def analyze_strategies(agents, stats):
    print("\n" + "=" * 60)
    print("ANALYSE DES STRATÉGIES APPRISES")
    print("=" * 60)
    
    # 1. Analyse globale
    final_coop = np.mean(stats['cooperation'][-50:])
    final_reward = np.mean(stats['rewards'][-50:])
    
    print(f"\n📊 RÉSULTATS GLOBAUX:")
    print(f"  • Taux de coopération final (50 derniers épisodes): {final_coop:.3f}")
    print(f"  • Récompense moyenne finale: {final_reward:.3f}")
    print(f"  • Epsilon final: {stats['epsilon'][-1]:.3f}")
    
    if final_coop > 0.7:
        print("  → FORTE COOPÉRATION ÉTABLIE! 🎉")
    elif final_coop > 0.4:
        print("  → Coopération modérée.")
    else:
        print("  → Défection dominante.")
    
    # 2. Analyse détaillée par agent
    print(f"\n🔍 ANALYSE PAR AGENT:")
    
    for i, agent in enumerate(agents[:3]):  # Analyser les 3 premiers
        print(f"\n🤖 AGENT {i}:")
        print("-" * 40)
        
        # Test avec des situations typiques
        test_cases = [
            ("Après C-C", [0, 0, 3, 3]),
            ("Après C-D (trahi)", [0, 1, 0, 5]),
            ("Après D-C (j'ai trahi)", [1, 0, 5, 0]),
            ("Après D-D", [1, 1, 1, 1]),
        ]
        
        for desc, state in test_cases:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = agent.policy_net(state_tensor)
                best_action = q_values.argmax().item()
                
                action_str = "COOPÉRER" if best_action == 0 else "TRAHIR"
                confidence = abs(q_values[0] - q_values[1])
                
                print(f"  {desc:25} → {action_str:10} "
                      f"(Q-C: {q_values[0]:.2f}, Q-D: {q_values[1]:.2f}, "
                      f"conf: {confidence:.2f})")
        
        print(f"  📈 Métriques agent:")
        print(f"    • Epsilon: {agent.epsilon:.3f}")
        print(f"    • Mémoire: {len(agent.memory)} expériences")
        print(f"    • Steps effectués: {agent.steps_done}")

# ==================== SAUVEGARDE JSON ====================
def save_training_summary(agents, stats, total_time, smoothed_coop):
    """Sauvegarde un résumé complet en JSON"""
    
    training_summary = {
        "project": "IPD with DQN - Baseline Model",
        "config": {
            "num_agents": config.NUM_AGENTS,
            "num_episodes": config.NUM_EPISODES,
            "steps_per_episode": config.STEPS_PER_EPISODE,
            "hidden_size": config.HIDDEN_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "gamma": config.GAMMA,
            "batch_size": config.BATCH_SIZE
        },
        "training_stats": {
            "total_time_seconds": total_time,
            "avg_cooperation_last_50": float(np.mean(stats['cooperation'][-50:])),
            "avg_reward_last_50": float(np.mean(stats['rewards'][-50:])),
            "final_epsilon": float(stats['epsilon'][-1]),
            "cooperation_trend": "increasing" if smoothed_coop.iloc[-1] > smoothed_coop.iloc[0] else "decreasing"
        },
        "agents_summary": {}
    }
    
    # Ajouter un résumé pour chaque agent
    for i, agent in enumerate(agents):
        # Échantillon de décisions
        decisions_sample = []
        for _ in range(5):  # 5 états aléatoires
            random_state = np.random.randn(4).tolist()
            with torch.no_grad():
                q_vals = agent.policy_net(torch.FloatTensor(random_state)).tolist()
                decisions_sample.append({
                    "state": random_state,
                    "q_cooperate": q_vals[0],
                    "q_defect": q_vals[1],
                    "predicted_action": "cooperate" if q_vals[0] > q_vals[1] else "defect"
                })
        
        training_summary["agents_summary"][f"agent_{i}"] = {
            "final_epsilon": float(agent.epsilon),
            "experience_memory_size": len(agent.memory),
            "total_steps": agent.steps_done,
            "decision_samples": decisions_sample
        }
    
    # Sauvegarder en JSON
    with open('ipd_training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Résumé sauvegardé dans 'ipd_training_summary.json'")

# ==================== RAPPORT TEXTE ====================
def generate_text_report(stats, total_time):
    """Génère un rapport texte simple"""
    
    with open('ipd_strategy_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RAPPORT DES STRATÉGIES APPRISES - IPD DQN\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CONFIGURATION DU MODÈLE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Nombre d'agents: {config.NUM_AGENTS}\n")
        f.write(f"Nombre d'épisodes: {config.NUM_EPISODES}\n")
        f.write(f"Steps par épisode: {config.STEPS_PER_EPISODE}\n")
        f.write(f"Taille du réseau: {config.HIDDEN_SIZE} neurones\n")
        f.write(f"Learning rate: {config.LEARNING_RATE}\n")
        f.write(f"Gamma: {config.GAMMA}\n\n")
        
        f.write("RÉSULTATS DE L'ENTRAÎNEMENT:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Temps total: {total_time:.1f} secondes\n")
        f.write(f"Taux de coopération final: {np.mean(stats['cooperation'][-50:]):.3f}\n")
        f.write(f"Récompense moyenne finale: {np.mean(stats['rewards'][-50:]):.3f}\n")
        f.write(f"Epsilon final: {stats['epsilon'][-1]:.3f}\n\n")
        
        f.write("INTERPRÉTATION:\n")
        f.write("-" * 40 + "\n")
        coop_rate = np.mean(stats['cooperation'][-50:])
        if coop_rate > 0.7:
            f.write("→ Les agents ont appris à coopérer de manière stable.\n")
            f.write("→ Cela suggère que dans cet environnement, la coopération\n")
            f.write("  est une stratégie plus profitable à long terme.\n")
        elif coop_rate > 0.4:
            f.write("→ Coopération modérée détectée.\n")
            f.write("→ Les agents alternent entre coopération et défection.\n")
        else:
            f.write("→ Défection dominante (équilibre de Nash classique).\n")
            f.write("→ Les agents n'ont pas trouvé d'avantage à coopérer.\n")
        
        f.write("\nFICHIERS GÉNÉRÉS:\n")
        f.write("-" * 40 + "\n")
        f.write("• ipd_complete_results.png : Graphiques d'entraînement\n")
        f.write("• ipd_training_summary.json : Résumé détaillé en JSON\n")
        f.write("• ipd_strategy_report.txt : Ce rapport\n")
    
    print(f"✅ Rapport texte généré: 'ipd_strategy_report.txt'")

# ==================== EXÉCUTION PRINCIPALE ====================
if __name__ == "__main__":
    # 1. Entraîner le modèle
    env, agents, stats, total_time = train_model()
    
    # 2. Visualiser les résultats
    smoothed_coop = plot_results(stats, total_time)
    
    # 3. Analyser les stratégies dans le terminal
    analyze_strategies(agents, stats)
    
    # 4. Sauvegarder en JSON (au lieu de .pth)
    save_training_summary(agents, stats, total_time, smoothed_coop)
    
    # 5. Générer un rapport texte
    generate_text_report(stats, total_time)
    
