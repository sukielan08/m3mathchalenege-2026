from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(42)

ACCOUNT_PROBS = {
    "gender": {"male": 0.46, "female": 0.28},
    "age": {"18-34": 0.52, "35-49": 0.39, "50-64": 0.25, "65+": 0.12},
    "edu": {"no_college": 0.27, "ba_plus": 0.44},
}

CHASE_PROBS = {
    "gender": {"male": 0.32, "female": 0.18},
    "age": {"18-34": 0.34, "35-49": 0.27, "50-64": 0.18, "65+": 0.10},
    "edu": {"no_college": 0.26, "ba_plus": 0.24},
}

BET_FREQ_WEIGHTS = {
    "monthly_or_less": 0.25,
    "every_3_4_weeks": 0.32,
    "weekly": 0.28,
    "multiple_per_week": 0.15,
}

DEPOSIT_FREQ_WEIGHTS = {
    "monthly": 0.40,
    "biweekly": 0.25,
    "weekly_plus": 0.35,
}

WITHDRAW_WEIGHTS = {
    "withdraw_often": 0.35,
    "leave_in_account": 0.45,
    "never_withdraw": 0.20,
}

WAGER_TIER_WEIGHTS = {
    "low": 0.50,
    "mid": 0.35,
    "high": 0.15,
}

P_BET_GIVEN_ACCOUNT = 0.83


def weighted_choice(weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs = probs / probs.sum()
    return str(RNG.choice(keys, p=probs))


def avg_prob(table: dict[str, dict[str, float]], profile: dict) -> float:
    return float(np.mean([
        table["gender"][profile["gender"]],
        table["age"][profile["age"]],
        table["edu"][profile["edu"]],
    ]))


def sample_account(profile: dict) -> bool:
    p_acc = avg_prob(ACCOUNT_PROBS, profile)
    return bool(RNG.random() < p_acc)


def sample_active_bettor() -> bool:
    return bool(RNG.random() < P_BET_GIVEN_ACCOUNT)


def sample_chases(profile: dict) -> bool:
    p = avg_prob(CHASE_PROBS, profile)
    return bool(RNG.random() < p)


def sample_behavior() -> dict:
    return {
        "bet_freq": weighted_choice(BET_FREQ_WEIGHTS),
        "deposit_freq": weighted_choice(DEPOSIT_FREQ_WEIGHTS),
        "withdraw": weighted_choice(WITHDRAW_WEIGHTS),
        "wager_tier": weighted_choice(WAGER_TIER_WEIGHTS),
    }


def sample_monthly_handle(wager_tier: str) -> float:
    if wager_tier == "low":
        return float(RNG.uniform(1, 100))
    if wager_tier == "mid":
        return float(RNG.uniform(101, 500))
    # high spend tier from report
    return float(RNG.lognormal(mean=np.log(900), sigma=0.8))


def handle_multiplier(deposit_freq: str, withdraw: str, chases: bool) -> float:
    m = 1.0
    if deposit_freq == "biweekly":
        m += 0.10
    elif deposit_freq == "weekly_plus":
        m += 0.20

    if withdraw == "leave_in_account":
        m += 0.10
    elif withdraw == "never_withdraw":
        m += 0.15

    if chases:
        m += 0.25

    return m


def frequency_points(bet_freq: str) -> int:
    if bet_freq == "weekly":
        return 1
    if bet_freq == "multiple_per_week":
        return 2
    return 0


def risk_score(profile: dict) -> int:
    score = 0
    score += 2 if profile["chases"] else 0
    score += 1 if profile["deposit_freq"] == "weekly_plus" else 0
    score += 1 if profile["withdraw"] in {"leave_in_account", "never_withdraw"} else 0
    score += frequency_points(profile["bet_freq"])
    return score


def risk_bucket(score: int) -> str:
    if score <= 1:
        return "low"
    if score <= 3:
        return "medium"
    return "high"


def hold_rate(risk: str) -> float:
    return {"low": 0.08, "medium": 0.10, "high": 0.15}[risk]


def outcome_std_fraction(risk: str) -> float:
    return {"low": 0.08, "medium": 0.15, "high": 0.30}[risk]


def simulate_profile(profile: dict) -> dict:
    prof = profile.copy()

    prof["has_account"] = sample_account(prof)
    if not prof["has_account"]:
        prof["placed_bet"] = False
        prof["annual_handle_est"] = 0.0
        prof["risk_score"] = 0
        prof["risk"] = "none"
        return prof

    prof["placed_bet"] = sample_active_bettor()
    if not prof["placed_bet"]:
        prof["annual_handle_est"] = 0.0
        prof["risk_score"] = 0
        prof["risk"] = "none"
        return prof

    behavior = sample_behavior()
    prof.update(behavior)
    prof["chases"] = sample_chases(prof)

    monthly_handle = sample_monthly_handle(prof["wager_tier"])
    m = handle_multiplier(prof["deposit_freq"], prof["withdraw"], prof["chases"])
    prof["annual_handle_est"] = 12 * monthly_handle * m

    prof["risk_score"] = risk_score(prof)
    prof["risk"] = risk_bucket(prof["risk_score"])
    return prof


def simulate_annual_net(profile: dict, n_trials: int = 8000) -> dict:
    annual_handle = float(profile["annual_handle_est"])
    risk = profile["risk"]
    score = int(profile["risk_score"])

    if annual_handle <= 0 or risk == "none":
        return {
            "risk": "none",
            "risk_score": 0,
            "hold_used": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "samples": np.zeros(n_trials),
        }

    h = hold_rate(risk)
    expected_loss = -h * annual_handle
    sigma = outcome_std_fraction(risk) * annual_handle

    samples = RNG.normal(loc=expected_loss, scale=sigma, size=n_trials)

    return {
        "risk": risk,
        "risk_score": score,
        "hold_used": h,
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p90": float(np.percentile(samples, 90)),
        "samples": samples,
    }


if __name__ == "__main__":
    example_profile = {
        "gender": "male",
        "age": "18-34",
        "edu": "ba_plus",
    }

    full_profile = simulate_profile(example_profile)
    print("=== Simulated Bettor Profile ===")
    for k, v in full_profile.items():
        if k != "samples":
            print(f"{k}: {v}")

    result = simulate_annual_net(full_profile, n_trials=8000)
    print("\n=== Annual Net Prediction ===")
    print(f"Risk tolerance: {result['risk']} (score {result['risk_score']})")
    print(f"Effective hold used: {result['hold_used']:.3f}")
    print(f"Annual handle estimate: ${full_profile['annual_handle_est']:,.0f}")
    print(f"Expected annual net (mean): ${result['mean']:,.0f}")
    print(f"Median annual net: ${result['median']:,.0f}")
    print(f"10th-90th percentile net: ${result['p10']:,.0f} to ${result['p90']:,.0f}")
