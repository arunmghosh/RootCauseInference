"""
Social Deduction Game Simulator (Solution Generator) - Phase III

This script simulates a social deduction game among covert operatives.
It outputs a full, unblinded play-by-play transcript ("solution generator")
showing player roles, investigation selections, traitor sabotage decisions,
and final win conditions.

Rules Summary:
- 5-9 players (5-7 players: 1 traitor; 8-9 players: 2 traitors).
- Every player gets 1 turn as investigation leader (total N investigations).
- On turn, leader selects 2 other players to join them (team of 3).
- No 3-player lineup set can be repeated across investigations.
- Regular agents always pass (succeed). Traitors can choose to pass or sabotage.
- Any sabotage in an investigation causes it to fail (sabotaged).
- Rule 1 (Mandatory Participation): Every player must take part in at least 2 investigations
  (including their own turn as leader). If any player is in < 2 investigations by game end,
  traitors automatically win.
- Rule 2 (Majority): Traitors win if majority of investigations failed.
  Ties (e.g. 3/6 or 4/8 sabotaged) mean loyal agents win.
"""

import random
import argparse
from dataclasses import dataclass
from typing import List, Set, Dict, Optional
from enum import Enum


class Role(Enum):
    AGENT = "Agent"
    TRAITOR = "Traitor"


class TraitorStrategy(Enum):
    RANDOM = "random"              # Randomly decide to sabotage with a probability
    ALWAYS_SABOTAGE = "always"      # Always sabotage whenever on an investigation
    SMART = "smart"                # Sabotage strategically (e.g. if close to winning majority)


@dataclass
class Player:
    id: int
    name: str
    role: Role
    participations: int = 0

    def is_traitor(self) -> bool:
        return self.role == Role.TRAITOR


@dataclass
class InvestigationResult:
    turn: int
    leader: Player
    operatives: List[Player]
    lineup_set: Set[int]
    sabotages: int
    success: bool
    traitor_decisions: Dict[str, str]  # Player name -> "SABOTAGE" or "PASSED"


@dataclass
class GameSummary:
    num_players: int
    num_traitors: int
    players: List[Player]
    investigations: List[InvestigationResult]
    mandatory_participation_fulfilled: bool
    unvisited_players: List[Player]
    total_sabotaged: int
    total_successful: int
    winner: Role
    win_reason: str


class GameSimulator:
    DEFAULT_NAMES = [
        "Agent Alpha", "Agent Bravo", "Agent Charlie", "Agent Delta", 
        "Agent Echo", "Agent Foxtrot", "Agent Golf", "Agent Hotel", "Agent India"
    ]

    def __init__(self, num_players: int = 6, seed: Optional[int] = None, 
                 traitor_strategy: str = "smart", sabotage_prob: float = 0.7,
                 player_names: Optional[List[str]] = None):
        if not (5 <= num_players <= 9):
            raise ValueError(f"Player count must be between 5 and 9. Received {num_players}.")
        
        self.num_players = num_players
        self.seed = seed
        self.rng = random.Random(seed)
        self.traitor_strategy = TraitorStrategy(traitor_strategy.lower())
        self.sabotage_prob = sabotage_prob

        # Determine traitors count: 1 for 5-7 players, 2 for 8-9 players
        self.num_traitors = 1 if num_players <= 7 else 2

        # Setup player names
        names = player_names or self.DEFAULT_NAMES[:num_players]
        if len(names) < num_players:
            names = [f"Operative {i+1}" for i in range(num_players)]

        # Secretly assign roles
        traitor_indices = set(self.rng.sample(range(num_players), self.num_traitors))
        self.players: List[Player] = [
            Player(
                id=i, 
                name=names[i], 
                role=Role.TRAITOR if i in traitor_indices else Role.AGENT
            )
            for i in range(num_players)
        ]

        self.used_lineups: Set[frozenset] = set()
        self.investigations: List[InvestigationResult] = []

    def _select_operatives(self, leader: Player, turn_idx: int) -> List[Player]:
        """
        Selects 2 other players to accompany the leader in an investigation team.
        Constraints:
        - Team of 3 (leader + 2 operatives) set must NOT have been used before.
        - Heuristic: Leaders try to select players who haven't participated yet if turns are running out.
        """
        candidates = [p for p in self.players if p.id != leader.id]

        # Sort candidates prioritizing those with lower participation counts
        shuffled_candidates = list(candidates)
        self.rng.shuffle(shuffled_candidates)
        shuffled_candidates.sort(key=lambda p: p.participations)

        # Try to find a valid pair (p1, p2) that forms an unused lineup
        valid_pairs = []
        for i in range(len(shuffled_candidates)):
            for j in range(i + 1, len(shuffled_candidates)):
                p1 = shuffled_candidates[i]
                p2 = shuffled_candidates[j]
                lineup = frozenset([leader.id, p1.id, p2.id])
                if lineup not in self.used_lineups:
                    under_participated_count = (1 if p1.participations < 2 else 0) + (1 if p2.participations < 2 else 0)
                    valid_pairs.append((under_participated_count, [p1, p2], lineup))

        if not valid_pairs:
            # Fallback search if heuristic didn't yield candidate
            all_candidates = list(candidates)
            for i in range(len(all_candidates)):
                for j in range(i + 1, len(all_candidates)):
                    p1 = all_candidates[i]
                    p2 = all_candidates[j]
                    lineup = frozenset([leader.id, p1.id, p2.id])
                    if lineup not in self.used_lineups:
                        valid_pairs.append((0, [p1, p2], lineup))

        if not valid_pairs:
            raise RuntimeError(f"Could not find any unused 3-player lineup for turn {turn_idx+1}.")

        # Sort valid pairs by under_participated count descending
        valid_pairs.sort(key=lambda item: item[0], reverse=True)
        chosen_pair = valid_pairs[0][1]
        chosen_lineup = valid_pairs[0][2]

        self.used_lineups.add(chosen_lineup)
        return chosen_pair

    def _decide_traitor_action(self, traitor: Player, team: List[Player], turn_idx: int) -> str:
        """Decides whether a traitor sabotages or plays along."""
        if self.traitor_strategy == TraitorStrategy.ALWAYS_SABOTAGE:
            return "SABOTAGE"
        elif self.traitor_strategy == TraitorStrategy.RANDOM:
            return "SABOTAGE" if self.rng.random() < self.sabotage_prob else "PASSED"
        elif self.traitor_strategy == TraitorStrategy.SMART:
            current_sabotages = sum(1 for inv in self.investigations if not inv.success)
            needed_sabotages = (self.num_players // 2) + 1 - current_sabotages
            turns_remaining = self.num_players - turn_idx

            if needed_sabotages <= 0:
                # Traitors already secured majority
                return "PASSED" if self.rng.random() < 0.5 else "SABOTAGE"
            
            if turns_remaining <= needed_sabotages:
                # Desperate: must sabotage whenever possible
                return "SABOTAGE"

            traitors_in_team = [p for p in team if p.is_traitor()]
            if len(traitors_in_team) > 1:
                # Avoid wasteful double-sabotage
                return "SABOTAGE" if self.rng.random() < 0.6 else "PASSED"

            return "SABOTAGE" if self.rng.random() < self.sabotage_prob else "PASSED"
        
        return "SABOTAGE"

    def run(self) -> GameSummary:
        """Executes the full game simulation."""
        start_idx = self.rng.randint(0, self.num_players - 1)
        turn_order = [(start_idx + i) % self.num_players for i in range(self.num_players)]

        for turn_num, leader_idx in enumerate(turn_order, start=1):
            leader = self.players[leader_idx]
            operatives = self._select_operatives(leader, turn_num - 1)
            team = [leader] + operatives

            for member in team:
                member.participations += 1

            traitor_decisions = {}
            sabotage_count = 0
            for member in team:
                if member.is_traitor():
                    decision = self._decide_traitor_action(member, team, turn_num - 1)
                    traitor_decisions[member.name] = decision
                    if decision == "SABOTAGE":
                        sabotage_count += 1
                else:
                    traitor_decisions[member.name] = "PASSED"

            success = (sabotage_count == 0)
            lineup_set = set(p.id for p in team)

            self.investigations.append(InvestigationResult(
                turn=turn_num,
                leader=leader,
                operatives=operatives,
                lineup_set=lineup_set,
                sabotages=sabotage_count,
                success=success,
                traitor_decisions=traitor_decisions
            ))

        # Check win conditions: Mandatory participation requires >= 2 investigations per player
        unvisited = [p for p in self.players if p.participations < 2]
        mandatory_fulfilled = (len(unvisited) == 0)

        total_sabotaged = sum(1 for inv in self.investigations if not inv.success)
        total_successful = sum(1 for inv in self.investigations if inv.success)
        majority_threshold = (self.num_players // 2) + 1

        if not mandatory_fulfilled:
            winner = Role.TRAITOR
            unvisited_names = ", ".join(f"{p.name} ({p.participations} inv)" for p in unvisited)
            win_reason = f"Mandatory participation rule violated! Player(s) [{unvisited_names}] were in fewer than 2 investigations. Traitors win automatically due to unfair work distribution."
        elif total_sabotaged >= majority_threshold:
            winner = Role.TRAITOR
            win_reason = f"Majority of investigations sabotaged ({total_sabotaged}/{self.num_players} failed, required {majority_threshold}). Traitors win!"
        else:
            winner = Role.AGENT
            win_reason = f"Loyal agents successfully completed {total_successful}/{self.num_players} investigations. Traitors failed to reach majority sabotage threshold ({total_sabotaged}/{self.num_players} failed). Loyal Agents win!"

        return GameSummary(
            num_players=self.num_players,
            num_traitors=self.num_traitors,
            players=self.players,
            investigations=self.investigations,
            mandatory_participation_fulfilled=mandatory_fulfilled,
            unvisited_players=unvisited,
            total_sabotaged=total_sabotaged,
            total_successful=total_successful,
            winner=winner,
            win_reason=win_reason
        )

    def print_play_by_play(self, summary: GameSummary) -> None:
        """Prints out the complete solution generator play-by-play transcript."""
        print("================================================================================")
        print("          SECRET OPERATIVE SOCIAL DEDUCTION GAME - PLAY-BY-PLAY SOLUTION        ")
        print("================================================================================\n")
        
        print("--- GAME SETUP & SECRET ROLES ---")
        print(f"Total Players: {summary.num_players}")
        print(f"Traitor Count: {summary.num_traitors}")
        print("Secret Role Assignments:")
        for p in summary.players:
            role_str = "TRAITOR" if p.is_traitor() else "LOYAL AGENT"
            print(f"  • {p.name:<15} : {role_str}")
        print()

        print("--- INVESTIGATION PLAY-BY-PLAY ---")
        for inv in summary.investigations:
            team_names = [inv.leader.name] + [op.name for op in inv.operatives]
            team_str = ", ".join(team_names)
            result_str = "SUCCESS (Evidence Found)" if inv.success else "SABOTAGED (Empty Investigation)"
            
            print(f"Turn {inv.turn}: Leader = {inv.leader.name}")
            print(f"  Lineup Team (3 Operatives): [{team_str}]")
            print("  Decisions:")
            for member_name, dec in inv.traitor_decisions.items():
                print(f"    - {member_name:<15}: {dec}")
            print(f"  Result: {result_str} (Sabotages = {inv.sabotages})")
            print("-" * 60)

        print("\n--- PARTICIPATION SUMMARY ---")
        for p in summary.players:
            print(f"  • {p.name:<15}: {p.participations} investigation(s)")
        print()

        print("--- GAME OUTCOME & SOLUTION REASONING ---")
        print(f"Total Successful Investigations : {summary.total_successful}")
        print(f"Total Sabotaged Investigations  : {summary.total_sabotaged}")
        print(f"Mandatory Participation Met     : {'YES' if summary.mandatory_participation_fulfilled else 'NO'}")
        print(f"\nWINNER: >>> {summary.winner.value.upper()}S WIN <<<")
        print(f"Reason: {summary.win_reason}")
        print("================================================================================")


def main():
    parser = argparse.ArgumentParser(description="Social Deduction Game Simulator (Solution Generator)")
    parser.add_argument("--players", type=int, default=6, choices=range(5, 10), help="Number of players (5-9)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--strategy", type=str, default="smart", choices=["smart", "random", "always"], help="Traitor sabotage strategy")
    parser.add_argument("--prob", type=float, default=0.7, help="Sabotage probability for random/smart strategy")

    args = parser.parse_args()

    simulator = GameSimulator(
        num_players=args.players,
        seed=args.seed,
        traitor_strategy=args.strategy,
        sabotage_prob=args.prob
    )
    summary = simulator.run()
    simulator.print_play_by_play(summary)


if __name__ == "__main__":
    main()
