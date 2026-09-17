#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>
#include <chrono>

// Sizing logic: Quarter Kelly criterion with a hard cap of 1.0% (0.01) of current bankroll
double quarter_kelly(double bankroll, double p, double q) {
    double b = q - 1.0;
    if (b <= 0.0) return 0.0;
    double ev = (p * q) - 1.0;
    if (ev <= 0.0) return 0.0;
    double f_star = ev / b;
    double f_quarter = f_star * 0.25;
    double fraction = std::min(f_quarter, 0.01);

    return bankroll * fraction;
}

struct Match {
    double odds[3];
    double probs[3];
    int actual_result;
};

struct PlayableBet {
    double odd;
    double prob;
    bool won;
};

int main(int argc, char* argv[]) {
    std::string filename = "backtest2.0.csv";
    if (argc > 1) {
        filename = argv[1];
    }
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Fallback for execution from project root
        filename = "backtest/backtest2.0.csv";
        file.open(filename);
    }
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << (argc > 1 ? argv[1] : "backtest2.0.csv") << "\n";
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::string line;
    std::vector<Match> matches;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string q_hstring, q_dstring, q_astring, p_hstring, p_dstring, p_astring, act_resstring;
        if (!std::getline(ss, q_hstring, ',')) continue;
        if (!std::getline(ss, q_dstring, ',')) continue;
        if (!std::getline(ss, q_astring, ',')) continue;
        if (!std::getline(ss, p_hstring, ',')) continue;
        if (!std::getline(ss, p_dstring, ',')) continue;
        if (!std::getline(ss, p_astring, ',')) continue;
        if (!std::getline(ss, act_resstring, ',')) continue;

        try {
            double q_h = std::stod(q_hstring);
            double q_d = std::stod(q_dstring);
            double q_a = std::stod(q_astring);
            double p_h = std::stod(p_hstring) / 100.0;
            double p_d = std::stod(p_dstring) / 100.0;
            double p_a = std::stod(p_astring) / 100.0;
            int res = std::stoi(act_resstring);

            matches.push_back({{q_h, q_d, q_a}, {p_h, p_d, p_a}, res});
        } catch (...) {
            continue;
        }
    }

    int total_bets = 0;
    int won_bets = 0;
    double peak = 1000.0;
    double max_drawdown = 0.0;
    double bankroll = 1000.0;
    double total_odds = 0.0;
    double flat_pnl = 0.0;
    std::vector<PlayableBet> playable_bets;

    // Filter rules: Odds in (1.6, 3.1), EV > 13%
    for (size_t i = 0; i < matches.size(); ++i) {
        double max_ev = 0.0;
        int best_index = -1;

        for (int j = 0; j < 3; ++j) {
            double ev = (matches[i].odds[j] * matches[i].probs[j]) - 1.0;
            if (ev > max_ev && matches[i].odds[j] > 1.6 && matches[i].odds[j] < 3.1) {
                max_ev = ev;
                best_index = j;
            }
        }

        if (best_index != -1 && max_ev > 0.13) {
            bool won = false;
            total_bets++;
            double prob = matches[i].probs[best_index];
            double odd = matches[i].odds[best_index];
            double stake = quarter_kelly(bankroll, prob, odd);

            bankroll -= stake;
            total_odds += odd;

            if (best_index == matches[i].actual_result) {
                bankroll += stake * odd;
                won_bets++;
                flat_pnl += (odd - 1.0);
                won = true;
            } else {
                flat_pnl -= 1.0;
                won = false;
            }

            playable_bets.push_back({odd, prob, won});
        }

        if (bankroll > peak) {
            peak = bankroll;
        } else {
            double dr = (peak - bankroll) / peak;
            max_drawdown = std::max(max_drawdown, dr);
        }
    }

    double avg_odds = (total_bets > 0) ? (total_odds / total_bets) : 0.0;
    double win_rate = (total_bets > 0) ? (static_cast<double>(won_bets) / total_bets * 100.0) : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "========================================================\n";
    std::cout << "       DEEP-PITCH C++ BACKTESTING & SIMULATION ENGINE   \n";
    std::cout << "========================================================\n";
    std::cout << "Matches Analyzed:        " << matches.size() << "\n";
    std::cout << "Bets Placed (EV > 13%):  " << total_bets << " (Won: " << won_bets << ", Win Rate: " << win_rate << "%)\n";
    std::cout << "Average Odds:            " << std::setprecision(3) << avg_odds << std::setprecision(2) << "\n";
    std::cout << "Initial Bankroll:        1000.00 EUR\n";
    std::cout << "Peak Bankroll:           " << peak << " EUR\n";
    std::cout << "Final Bankroll:          " << bankroll << " EUR (" << ((bankroll - 1000.0) / 10.0) << "% ROI)\n";
    std::cout << "Max Drawdown:            " << (max_drawdown * 100.0) << "%\n";
    std::cout << "Flat PnL:                " << flat_pnl << " units\n";

    // ==========================================
    // Monte Carlo Simulation Setup (10,000 runs)
    // ==========================================
    const int NUM_SIMULATIONS = 10000;
    std::vector<size_t> indices(playable_bets.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    std::vector<double> final_bankrolls;
    std::vector<double> max_drawdowns;
    final_bankrolls.reserve(NUM_SIMULATIONS);
    max_drawdowns.reserve(NUM_SIMULATIONS);

    int ruin_count = 0;

    for (int sim = 0; sim < NUM_SIMULATIONS; ++sim) {
        std::shuffle(indices.begin(), indices.end(), rng);

        double sim_bankroll = 1000.0;
        double sim_peak = 1000.0;
        double sim_max_dd = 0.0;

        for (size_t k = 0; k < indices.size(); ++k) {
            const auto& bet = playable_bets[indices[k]];

            double stake = quarter_kelly(sim_bankroll, bet.prob, bet.odd);
            sim_bankroll -= stake;

            if (bet.won) {
                sim_bankroll += stake * bet.odd;
            }

            if (sim_bankroll > sim_peak) {
                sim_peak = sim_bankroll;
            } else {
                double dr = (sim_peak - sim_bankroll) / sim_peak;
                sim_max_dd = std::max(sim_max_dd, dr);
            }

            // Critical ruin threshold (< 200 EUR = 80% loss)
            if (sim_bankroll < 200.0) {
                ruin_count++;
                break;
            }
        }

        final_bankrolls.push_back(sim_bankroll);
        max_drawdowns.push_back(sim_max_dd);
    }

    std::sort(final_bankrolls.begin(), final_bankrolls.end());
    std::sort(max_drawdowns.begin(), max_drawdowns.end());

    double ruin_prob = (static_cast<double>(ruin_count) / NUM_SIMULATIONS) * 100.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "\n========================================================\n";
    std::cout << "       MONTE CARLO ROBUSTNESS SIMULATION (10,000 RUNS)   \n";
    std::cout << "========================================================\n";
    std::cout << "Probability of Ruin (< 200 EUR): " << ruin_prob << "%\n\n";

    std::cout << "--- FINAL BANKROLL DISTRIBUTION ---\n";
    std::cout << "Pessimistic (5th percentile):   " << final_bankrolls[static_cast<size_t>(NUM_SIMULATIONS * 0.05)] << " EUR\n";
    std::cout << "Median      (50th percentile):  " << final_bankrolls[static_cast<size_t>(NUM_SIMULATIONS * 0.50)] << " EUR\n";
    std::cout << "Optimistic  (95th percentile):  " << final_bankrolls[static_cast<size_t>(NUM_SIMULATIONS * 0.95)] << " EUR\n\n";

    std::cout << "--- MAX DRAWDOWN DISTRIBUTION ---\n";
    std::cout << "Optimistic  (5th percentile):   " << max_drawdowns[static_cast<size_t>(NUM_SIMULATIONS * 0.05)] * 100.0 << "%\n";
    std::cout << "Median      (50th percentile):  " << max_drawdowns[static_cast<size_t>(NUM_SIMULATIONS * 0.50)] * 100.0 << "%\n";
    std::cout << "Pessimistic (95th percentile):  " << max_drawdowns[static_cast<size_t>(NUM_SIMULATIONS * 0.95)] * 100.0 << "%\n";
    std::cout << "Worst Absolute Drawdown:        " << max_drawdowns.back() * 100.0 << "%\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Simulation Execution Time: " << elapsed_ms << " ms\n";
    std::cout << "========================================================\n";

    return 0;
}
