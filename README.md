# Gambling Impact Model (MathWorks M3 Challenge) - Top 18% of Teams in the Nation

This project analyzes the economic and behavioral impact of online sports betting using mathematical modeling and simulation.

The work was developed for the 2026 MathWorks Math Modeling Challenge.

## Overview

The project builds three models:

1. Disposable Income Model
   Estimates disposable income using income, taxes, and essential expenditures across demographics. It as income remaining after taxes and "essential expenditures" (housing, food, healthcare, etc.). It accounts for 2024 progressive tax brackets in both the US and UK.

2. Gambling Behavior Simulation
   Uses probabilistic modeling and Monte Carlo simulation to estimate annual betting losses. Used a risk-scoring system to calibrate "hold rates" based on behaviors like "chasing losses" and deposit frequency.

3. Macroeconomic Impact Model
   Evaluates the effect of sports betting on national GDP through tax revenue, industry production, and consumer spending substitution. It uses a "consumer substitution rate" ($\lambda$) to determine how much gambling spending replaces other economic consumption.
   
## Methods

Key techniques used:

- probabilistic modeling
- Monte Carlo simulation
- lognormal wager distributions
- demographic parameter sensitivity analysis
- fiscal multiplier modeling

## Tools

Python  
NumPy  
Matplotlib

## Results

The models suggest that while sports betting generates government tax revenue and industry output, aggregate consumer losses exceed the macroeconomic benefits under realistic assumptions.

## Repository Structure

src/ – core modeling code  
paper/ – final research paper  
figures/ – graphs and model outputs

## Author

Sukirtthan Elanjchezhiyan, Varun Kaldindi, Ethan Manuel, Ameya Tyagi, Shreyash Koli
