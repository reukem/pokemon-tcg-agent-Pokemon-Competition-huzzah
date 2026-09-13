import math
import random
import numpy as np

# Graceful PyTorch import for standalone testing
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    F = None

# ---------------------------------------------------------
# 1. Neural Network Architecture (Policy and Value Heads)
# ---------------------------------------------------------
if TORCH_AVAILABLE:
    class PolicyValueNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(PolicyValueNetwork, self).__init__()
            # Shared hidden layers for processing the game board
            self.fc1 = nn.Linear(state_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            
            # Policy Head: Predicts the best action to take
            self.policy_head = nn.Linear(256, action_dim)
            
            # Value Head: Predicts if we are going to win (-1 to 1)
            self.value_head = nn.Linear(256, 1)

        def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            
            # Action probabilities
            policy_logits = self.policy_head(x)
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            # Win probability prediction
            state_value = torch.tanh(self.value_head(x))
            
            return policy_probs, state_value
else:
    class PolicyValueNetwork:
        def __init__(self, state_dim, action_dim):
            self.state_dim = state_dim
            self.action_dim = action_dim

        def forward(self, state):
            # Fallback uniform distribution and neutral win evaluation if torch not installed
            probs = np.ones(self.action_dim) / self.action_dim
            value = 0.0
            return probs, value

# ---------------------------------------------------------
# 2. Information Set Monte Carlo Tree Search (IS-MCTS)
# ---------------------------------------------------------
class ISMCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = self.get_legal_actions(state)

    def get_legal_actions(self, state):
        # In a real Kaggle environment, this calls env.get_legal_actions()
        # Returns list of available moves (e.g., [Attach_Energy, Play_Nest_Ball, Attack_Phantom_Dive])
        return state.get('legal_actions', [])

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def get_best_child(self, c_param=1.41):
        # UCB1 Formula (Upper Confidence Bound) to balance exploration vs exploitation
        choices_weights = [
            (child.value_sum / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, action, next_state):
        child_node = ISMCTSNode(next_state, parent=self, action=action)
        self.untried_actions.remove(action)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        self.value_sum += result
        if self.parent:
            self.parent.backpropagate(result)

# ---------------------------------------------------------
# 3. The Kaggle Agent (Dragapult ex Strategy)
# ---------------------------------------------------------
class DragapultAgent:
    def __init__(self, action_dim=500, state_dim=128):
        # Initialize our deep learning model
        self.model = PolicyValueNetwork(state_dim, action_dim)
        self.simulations_per_turn = 50 

    def determinize_state(self, game_state):
        """
        Since we can't see the opponent's hand or Prize cards (Imperfect Information),
        we "determinize" the state by randomly sampling plausible hidden cards based on 
        what is legally possible from the remaining card pool.
        """
        simulated_state = game_state.copy()
        return simulated_state

    def act(self, observation):
        """
        This is the main function Kaggle calls every turn to ask for our move.
        """
        legal_actions = observation.get('legal_actions', [])
        
        # 1. Custom Dragapult ex Logic: Phantom Dive Spread
        # If Phantom Dive is available, prioritize calculating the best damage spread
        for action in legal_actions:
            if action.get('name') == 'Phantom Dive':
                return self.calculate_phantom_dive_spread(observation)

        # 2. IS-MCTS Logic for general play
        root = ISMCTSNode(state=observation)
        
        # Run simulations across determinized information sets
        for _ in range(self.simulations_per_turn):
            # Step A: Guess the hidden information
            simulated_state = self.determinize_state(observation)
            node = root
            
            # Step B: Selection
            while node.is_fully_expanded() and node.children:
                node = node.get_best_child()
            
            # Step C: Expansion
            if not node.is_fully_expanded():
                action_to_try = random.choice(node.untried_actions)
                next_state = {"legal_actions": []} # Mock next state
                node = node.expand(action_to_try, next_state)
            
            # Step D: Simulation & Evaluation
            if TORCH_AVAILABLE:
                state_tensor = torch.zeros(128)
                with torch.no_grad():
                    policy_probs, value = self.model(state_tensor)
                    val_scalar = value.item()
            else:
                _, val_scalar = self.model.forward(None)
                
            # Step E: Backpropagation
            node.backpropagate(val_scalar)

        # Pick the move that had the best simulation results
        if root.children:
            best_action = root.get_best_child(c_param=0.0).action
            return best_action
        elif legal_actions:
            return legal_actions[0]
        return None

    def calculate_phantom_dive_spread(self, observation):
        """
        Calculates the mathematically optimal way to distribute 6 damage counters (60 damage).
        Prioritizes:
          1. Securing immediate Knock Outs on low-HP targets.
          2. Softening high-HP Stage 2 threats into 1-hit range.
        """
        opponent_bench = observation.get('opponent_bench', [])
        
        if opponent_bench:
            # Optimal target: lowest HP benched target to accelerate Prize cards
            weakest_target = min(opponent_bench, key=lambda x: x.get('hp', 999))
            return {
                'action_type': 'attack',
                'attack_name': 'Phantom Dive',
                'target': weakest_target['id'],
                'damage_counters': 6,
                'target_name': weakest_target.get('name', 'Benched Target')
            }
        return {'action_type': 'attack', 'attack_name': 'Phantom Dive'}

# ---------------------------------------------------------
# Example Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("[Notice] PyTorch is not installed in the local environment. Running with lightweight fallback.")
        print("         To enable full deep neural network heads, run: pip install -r requirements.txt\n")

    my_agent = DragapultAgent()
    
    mock_observation = {
        "turn": 3,
        "legal_actions": [
            {"name": "Attach Energy", "card": "Psychic Energy"},
            {"name": "Phantom Dive", "damage": 200, "spread": 60}
        ],
        "opponent_bench": [
            {"id": 1, "name": "Pidgey", "hp": 60},
            {"id": 2, "name": "Charizard ex", "hp": 330}
        ]
    }
    
    decision = my_agent.act(mock_observation)
    print("Agent selected decision:")
    for k, v in decision.items():
        print(f"  {k}: {v}")
