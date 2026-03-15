from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# US TAX + DISPOSABLE INCOME MODEL
# ============================================================

US_FED_STD_DED_SINGLE = 14600

US_FED_BRACKETS_SINGLE_2024 = [
    (11600, 0.10),
    (47150, 0.12),
    (100525, 0.22),
    (191950, 0.24),
    (243725, 0.32),
    (609350, 0.35),
    (float("inf"), 0.37),
]

US_SS_CAP_2024 = 168600
US_ADD_MED_THRESH_SINGLE_2024 = 200000

US_NO_TAX_STATES = {"AK", "FL", "NV", "SD", "TN", "TX", "WA", "WY"}

US_FLAT_STATES_2024 = {
    "IL": 0.0495,
    "CO": 0.0463,
    "PA": 0.0307,
    "MI": 0.0425,
    "IN": 0.0323,
    "UT": 0.0495,
}

US_PROGRESSIVE_2024 = {
    "CA": [
        (9325, 0.01),
        (22107, 0.02),
        (34892, 0.04),
        (48435, 0.06),
        (61214, 0.08),
        (312686, 0.093),
        (375221, 0.103),
        (625369, 0.113),
        (float("inf"), 0.123),
    ],
    "NY": [
        (17150, 0.04),
        (23600, 0.045),
        (27900, 0.0525),
        (43000, 0.059),
        (161550, 0.0621),
        (323200, 0.0685),
        (2155350, 0.0965),
        (5000000, 0.103),
        (float("inf"), 0.109),
    ],
}

US_STATE_TO_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest", "WI": "Midwest",
    "IA": "Midwest", "KS": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "SD": "Midwest",
    "AL": "South", "AR": "South", "DE": "South", "FL": "South", "GA": "South", "KY": "South",
    "LA": "South", "MD": "South", "MS": "South", "NC": "South", "OK": "South", "SC": "South",
    "TN": "South", "TX": "South", "VA": "South", "WV": "South",
    "AK": "West", "AZ": "West", "CA": "West", "CO": "West", "HI": "West", "ID": "West",
    "MT": "West", "NV": "West", "NM": "West", "OR": "West", "UT": "West", "WA": "West", "WY": "West",
}

US_MU_ALL = 25000
US_MU_INCOME = {"<30k": 19000, "30-60k": 22000, "60-100k": 26000, "100-150k": 32000, "150k+": 40000}
US_MU_AGE = {"Under25": 21000, "25-34": 23000, "35-44": 25000, "45-54": 27000, "55-64": 28000, "65-74": 27500, "75+": 26500}
US_MU_REGION = {"Northeast": 29000, "Midwest": 24500, "South": 23500, "West": 28000}
US_MU_EDU = {"HS_or_less": 22000, "Some_college": 25000, "Bachelors": 29000, "Graduate": 32000}
US_MU_EARNERS = {"1": 23000, "2": 27000, "3+": 30000}

# ============================================================
# UK TAX + DISPOSABLE INCOME MODEL
# ============================================================

UK_NI_PT = 12570.0
UK_NI_UEL = 50270.0

UK_MU_ALL = 22000
UK_MU_INCOME = {"<30k": 17000, "30-60k": 20500, "60-100k": 25000, "100-150k": 30500, "150k+": 38000}
UK_MU_AGE = {"Under25": 18500, "25-34": 20500, "35-44": 22500, "45-54": 24000, "55-64": 24500, "65-74": 24000, "75+": 23000}
UK_MU_REGION = {"England": 22500, "Wales": 21500, "Scotland": 22000, "Northern Ireland": 21000}
UK_MU_EDU = {"HS_or_less": 20000, "Some_college": 22000, "Bachelors": 24500, "Graduate": 26500}
UK_MU_EARNERS = {"1": 20500, "2": 23500, "3+": 25500}


def us_federal_tax_2024_single(income: float) -> float:
    taxable = max(0.0, income - US_FED_STD_DED_SINGLE)
    tax = 0.0
    lower = 0.0

    for upper, rate in US_FED_BRACKETS_SINGLE_2024:
        if taxable <= lower:
            break
        chunk = min(taxable, upper) - lower
        tax += chunk * rate
        lower = upper

    return tax


def us_payroll_tax_2024(income: float) -> float:
    ss = 0.062 * min(income, US_SS_CAP_2024)
    medicare = 0.0145 * income
    add_medicare = 0.009 * max(0.0, income - US_ADD_MED_THRESH_SINGLE_2024)
    return ss + medicare + add_medicare


def us_state_tax_2024(income: float, state: str) -> float:
    state = state.upper()

    if state in US_NO_TAX_STATES:
        return 0.0

    if state in US_FLAT_STATES_2024:
        return income * US_FLAT_STATES_2024[state]

    if state in US_PROGRESSIVE_2024:
        tax = 0.0
        lower = 0.0
        for upper, rate in US_PROGRESSIVE_2024[state]:
            if income <= lower:
                break
            chunk = min(income, upper) - lower
            tax += chunk * rate
            lower = upper
        return tax

    return 0.0


def after_tax_income_us(income: float, state: str) -> float:
    return income - (
        us_federal_tax_2024_single(income)
        + us_payroll_tax_2024(income)
        + us_state_tax_2024(income, state)
    )


def uk_personal_allowance(income: float) -> float:
    allowance = 12570.0
    if income > 100000:
        allowance = max(0.0, 12570.0 - (income - 100000) / 2.0)
    return allowance


def uk_income_tax_eng_2024_25(income: float) -> float:
    pa = uk_personal_allowance(income)
    taxable = max(0.0, income - pa)

    basic_width = max(0.0, 50270.0 - pa)
    higher_width = 125140.0 - 50270.0

    tax = 0.0
    basic = min(taxable, basic_width)
    tax += 0.20 * basic

    remaining = taxable - basic
    higher = min(max(0.0, remaining), higher_width)
    tax += 0.40 * higher

    remaining2 = remaining - higher
    if remaining2 > 0:
        tax += 0.45 * remaining2

    return tax


def uk_employee_ni_2024_25(income: float) -> float:
    if income <= UK_NI_PT:
        return 0.0
    main_band = min(income, UK_NI_UEL) - UK_NI_PT
    above = max(0.0, income - UK_NI_UEL)
    return 0.08 * main_band + 0.02 * above


def after_tax_income_uk(income: float) -> float:
    return income - (uk_income_tax_eng_2024_25(income) + uk_employee_ni_2024_25(income))


def us_age_group(age: int) -> str:
    if age < 25:
        return "Under25"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    if age < 75:
        return "65-74"
    return "75+"


def income_group(income: float) -> str:
    if income < 30000:
        return "<30k"
    if income < 60000:
        return "30-60k"
    if income < 100000:
        return "60-100k"
    if income < 150000:
        return "100-150k"
    return "150k+"


def earners_group(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def essentials_hat_us(income: float, age: int, state: str, education: str, earners: int) -> float:
    g_inc = income_group(income)
    g_age = us_age_group(age)
    g_reg = US_STATE_TO_REGION.get(state.upper(), "Midwest")
    g_ear = earners_group(earners)

    base = US_MU_INCOME[g_inc]
    adj = (
        (US_MU_AGE[g_age] - US_MU_ALL)
        + (US_MU_REGION[g_reg] - US_MU_ALL)
        + (US_MU_EDU[education] - US_MU_ALL)
        + (US_MU_EARNERS[g_ear] - US_MU_ALL)
    )
    return base + adj


def essentials_hat_uk(income: float, age: int, nation: str, education: str, earners: int) -> float:
    g_inc = income_group(income)
    g_age = us_age_group(age)
    g_nat = nation
    g_ear = earners_group(earners)

    base = UK_MU_INCOME[g_inc]
    adj = (
        (UK_MU_AGE[g_age] - UK_MU_ALL)
        + (UK_MU_REGION[g_nat] - UK_MU_ALL)
        + (UK_MU_EDU[education] - UK_MU_ALL)
        + (UK_MU_EARNERS[g_ear] - UK_MU_ALL)
    )
    return base + adj


def disposable_income_us(income: float, age: int, state: str, education: str = "Some_college", earners: int = 1) -> float:
    net = after_tax_income_us(income, state)
    ess = essentials_hat_us(income, age, state, education, earners)
    return max(0.0, net - ess)


def disposable_income_uk(income: float, age: int, nation: str, education: str = "Some_college", earners: int = 1) -> float:
    net = after_tax_income_uk(income)
    ess = essentials_hat_uk(income, age, nation, education, earners)
    return max(0.0, net - ess)


def disposable_income(country: str, income: float, age: int, place: str, education: str = "Some_college", earners: int = 1) -> float:
    c = country.upper().strip()
    if c == "US":
        return disposable_income_us(income, age, place, education=education, earners=earners)
    if c == "UK":
        return disposable_income_uk(income, age, place, education=education, earners=earners)
    raise ValueError("country must be 'US' or 'UK'")


def plot_disposable_income_vs_gross() -> None:
    incomes = np.linspace(20000, 200000, 30)

    us_vals = [
        disposable_income("US", float(inc), 40, "IL", education="Bachelors", earners=2)
        for inc in incomes
    ]
    uk_vals = [
        disposable_income("UK", float(inc), 40, "England", education="Bachelors", earners=2)
        for inc in incomes
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(incomes, us_vals, label="US (IL, age 40)")
    plt.plot(incomes, uk_vals, label="UK (England, age 40)")
    plt.xlabel("Gross Income (annual)")
    plt.ylabel("Disposable Income (annual)")
    plt.title("US vs UK: Disposable Income vs Gross Income")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_us_tornado_sensitivity() -> None:
    base = {
        "country": "US",
        "income": 85000,
        "age": 40,
        "place": "IL",
        "education": "Bachelors",
        "earners": 2,
    }

    base_di = disposable_income(**base)

    scenarios = [
        ("Age 25", ("US", 85000, 25, "IL", "Bachelors", 2)),
        ("Age 70", ("US", 85000, 70, "IL", "Bachelors", 2)),
        ("State TX", ("US", 85000, 40, "TX", "Bachelors", 2)),
        ("State CA", ("US", 85000, 40, "CA", "Bachelors", 2)),
        ("Edu HS", ("US", 85000, 40, "IL", "HS_or_less", 2)),
        ("Edu Grad", ("US", 85000, 40, "IL", "Graduate", 2)),
        ("Earners 1", ("US", 85000, 40, "IL", "Bachelors", 1)),
        ("Earners 3", ("US", 85000, 40, "IL", "Bachelors", 3)),
    ]

    labels = []
    deltas = []

    for name, args in scenarios:
        di = disposable_income(*args)
        labels.append(name)
        deltas.append(di - base_di)

    order = np.argsort(np.abs(deltas))[::-1]
    labels = [labels[i] for i in order]
    deltas = [deltas[i] for i in order]

    plt.figure(figsize=(8, 5))
    plt.barh(labels, deltas)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Δ Disposable Income vs Baseline")
    plt.title("US Tornado Sensitivity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(disposable_income("US", 85000, 40, "IL", education="Bachelors", earners=2))
    print(disposable_income("UK", 52000, 22, "England", education="Bachelors", earners=2))
    plot_disposable_income_vs_gross()
    plot_us_tornado_sensitivity()
