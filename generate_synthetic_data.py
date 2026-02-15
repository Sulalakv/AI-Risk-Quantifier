import pandas as pd
import numpy as np

np.random.seed(42)

N = 5000  # number of rows to generate

# --- Synthetic Column Generation ---

Duration = np.random.randint(5, 60, N)                     # weeks
Team_Size = np.random.randint(3, 30, N)                    # number of developers
Budget = np.random.randint(30000, 300000, N)               # dollars
Past_Delays = np.random.randint(0, 10, N)                  # count
Tech_Complexity = np.random.randint(1, 6, N)               # scale 1–5

# --- Risk Level Logic (Realistic Rules) ---

Risk_Level = []

for i in range(N):
    d = Duration[i]
    t = Team_Size[i]
    b = Budget[i]
    p = Past_Delays[i]
    c = Tech_Complexity[i]

    score = 0

    # Duration
    if d > 45: score += 2
    elif d > 30: score += 1

    # Complexity
    if c >= 4: score += 2
    elif c == 3: score += 1

    # Past delays
    if p >= 5: score += 2
    elif p >= 2: score += 1

    # Budget vs team (low budget & large team = high risk)
    if b < 80000 and t > 15:
        score += 2

    # Large budgets reduce risk a bit
    if b > 200000: 
        score -= 1

    # Final risk rules
    if score >= 4:
        Risk_Level.append("High")
    elif score >= 2:
        Risk_Level.append("Medium")
    else:
        Risk_Level.append("Low")

# --- Create DataFrame ---
df = pd.DataFrame({
    "Duration": Duration,
    "Team_Size": Team_Size,
    "Budget": Budget,
    "Past_Delays": Past_Delays,
    "Tech_Complexity": Tech_Complexity,
    "Risk_Level": Risk_Level
})

# Save CSV
df.to_csv("SPARQ_training_data_5000.csv", index=False)

print("Generated 5000 synthetic rows → SPARQ_training_data_5000.csv")
