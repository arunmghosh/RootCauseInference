# Can AI Solve Mysteries? 
I trained a DQN to play a card game that I developed called *Signal12*. The game has four players divided into two teams, and features public signals between teammates to test communication. The DQN achieves this with **two independently-trained models** — one for signaling, one for card play — each trained in its *own* self-play games against a deterministic scripted bot. This experiment showed that Signal12 has a large component of luck, which created too much noise for the signaling mechanism to significantly improve the DQN's performance. Nonetheless, the win rate was still consistently over 50% (sometimes over 60%), which is mild evidence of learning. 

[See "AI Mystery Solver" Powerpoint for more information]

## Overview


## Results
 

## Approach

### 1. Data Formulation
   

### 2. Model Architecture
  

### 3. Training & Evaluation
 

## What "Signal12" Looks Like


## Usage

