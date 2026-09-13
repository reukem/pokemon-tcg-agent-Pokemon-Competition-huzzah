import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Neural Network Architecture (Policy and Value Heads)
# ---------------------------------------------------------
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
        self.simulations_per_turn = 100 

    def determinize_state(self, game_state):
        """
        Since we can't see the opponent's hand or Prize cards (Imperfect Information),
        we "determinize" the state by randomly guessing their hidden cards based on 
        what is legally possible.
        """
        simulated_state = game_state.copy()
        # Logic to randomly fill opponent's hand from the remaining unknown cards
        # simulated_state['opponent_hand'] = random.sample(unknown_cards, size=hand_size)
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
        
        # Run thousands of simulations in our head before making a move
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
                # In real code: next_state = KaggleSimulator.step(simulated_state, action_to_try)
                next_state = {"legal_actions": []} # Mock next state
                node = node.expand(action_to_try, next_state)
            
            # Step D: Simulation & Neural Net Evaluation
            # Convert state to tensor for Neural Network
            state_tensor = torch.zeros(128) # Mock tensor
            with torch.no_grad():
                policy_probs, value = self.model(state_tensor)
                
            # Step E: Backpropagation
            node.backpropagate(value.item())

        # Finally, pick the move that had the best simulation results
        best_action = root.get_best_child(c_param=0.0).action
        return best_action

    def calculate_phantom_dive_spread(self, observation):
        """
        Calculates the mathematically optimal way to distribute 6 damage counters.
        """
        opponent_bench = observation.get('opponent_bench', [])
        
        # Very simple heuristic: target the weakest Pokemon to get fast Prize cards
        if opponent_bench:
            weakest_target = min(opponent_bench, key=lambda x: x.get('hp', 999))
            return {
                'action_type': 'attack',
                'attack_name': 'Phantom Dive',
                'target': weakest_target['id'],
                'counters': 6
            }
        return {'action_type': 'attack', 'attack_name': 'Phantom Dive'}

# ---------------------------------------------------------
# Example of how Kaggle runs this
# ---------------------------------------------------------
if __name__ == "__main__":
    # Initialize the agent
    my_agent = DragapultAgent()
    
    # Mock observation from Kaggle
    mock_observation = {
        "turn": 3,
        "legal_actions": [
            {"name": "Attach Energy", "card": "Psychic Energy"},
            {"name": "Phantom Dive", "damage": 200, "spread": 60}
        ],
        "opponent_bench": [{"id": 1, "hp": 60}, {"id": 2, "hp": 220}]
    }
    
    # Get the AI's decision
    decision = my_agent.act(mock_observation)
    print("AI decided to:", decision)
