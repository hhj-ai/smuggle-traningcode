import math

class RewardCalculator:
    """
    Implements the core AURORA reward dynamics.
    """
    def __init__(self):
        pass

    def calculate_vlm_reward(self, correct_claims, incorrect_claims, total_claims, description_length):
        """
        Reward for the Generator (VLM).
        Goal: Maximize correct claims, minimize incorrect ones.
        """
        if total_claims == 0:
            return 0.0

        # Accuracy Component
        # Range: -1.0 to 1.0
        accuracy_score = (correct_claims - incorrect_claims) / total_claims
        
        # Bonus for density of correct information
        # Log scaling prevents runaway rewards for long lists
        bonus_score = 0.5 * math.log(correct_claims + 1)
        
        # Length Penalty (Sigmoid or Exponential)
        # Penalize descriptions < 20 words
        length_penalty = 0.0
        if description_length < 20:
            length_penalty = -2.0
            
        total = accuracy_score + bonus_score + length_penalty
        return total

    def calculate_verifier_reward(self, claim_status, claim_traceable, correlation_score=0.0):
        """
        Reward for the Verifier (Opponent).
        Goal: Find VLM errors (+5) or Verify Truth (+1).
        Penalize: Hallucinating claims (-10) or being Useless (-0.1).
        
        Added: Correlation Penalty to prevent redundancy.
        """
        reward = 0.0
        
        # 1. Traceability Check (Critical!)
        # If the Verifier makes up a claim that the VLM never said, punish heavily.
        if not claim_traceable:
            return -10.0
            
        # 2. Attack Success (The "Opponent" Logic)
        if claim_status == 'incorrect':
            # High reward: You caught the VLM lying!
            reward = 5.0
            
        # 3. Verification Success
        elif claim_status == 'correct':
            # Small reward: You confirmed a fact.
            reward = 1.0
            
        # 4. Uncertainty / Uselessness
        elif claim_status == 'uncertain':
            # Penalty: You asked a question the tools couldn't answer.
            # Idea says -0.1, adjusted here
            reward = -0.1
            
        # 5. Correlation Penalty (New from Idea)
        # Reduce reward if the claim is too similar to others (redundant)
        # correlation_score is usually between 0 and 1.
        # Weighting it by 2.0 to be significant.
        reward -= (correlation_score * 2.0)
            
        return reward
