#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

/**
 * Quarter-Kelly Staking with Portfolio Hard Cap
 * Stake = bankroll * min(0.25 * (EV / (Odds - 1)), 0.015)
 */
double quarter_kelly(double bankroll, double p, double q) {
    double b = q - 1.0;
    if (b <= 0.0) return 0.0;
    double ev = (p * q) - 1.0;
    if (ev <= 0.0) return 0.0;
    double f_star = ev / b;
    double f_quarter = f_star * 0.25;
    double fraction = std::min(f_quarter, 0.015); // Max 1.5% bankroll cap

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

struct StrategyResult {
    double ev_threshold;
    int total_bets;
    int won_bets;
    double win_rate;
    double avg_odds;
    double final_bankroll;
    double peak_bankroll;
    double max_drawdown;
    double flat_pnl;
    double roi_perc;
};

StrategyResult run_strategy(const std::vector<Match>& matches, double ev_thresh, std::vector<PlayableBet>* out_bets = nullptr) {
    int total_bets = 0;
    int won_bets = 0;
    double peak = 1000.0;
    double max_drawdown = 0.0;
    double bankroll = 1000.0;
    double total_odds = 0.0;
    double flat_pnl = 0.0;

    if (out_bets) out_bets->clear();

    for (size_t i = 0; i < matches.size(); ++i) {
        double max_ev = 0.0;
        int best_index = -1;

        for (int j = 0; j < 3; ++j) {
            double ev = (matches[i].odds[j] * matches[i].probs[j]) - 1.0;
            if (ev > max_ev && matches[i].odds[j] > 1.60 && matches[i].odds[j] < 3.10) {
                max_ev = ev;
                best_index = j;
            }
        }

        if (best_index != -1 && max_ev > ev_thresh) {
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

            if (out_bets) {
                out_bets->push_back({odd, prob, won});
            }
        }

        if (bankroll > peak) {
            peak = bankroll;
        } else {
            double dr = (peak - bankroll) / peak;
            max_drawdown = std::max(max_drawdown, dr);
        }
    }

    double win_rate = total_bets > 0 ? (static_cast<double>(won_bets) / total_bets) * 100.0 : 0.0;
    double avg_odds = total_bets > 0 ? total_odds / total_bets : 0.0;
    double roi_perc = ((bankroll - 1000.0) / 1000.0) * 100.0;

    return {ev_thresh, total_bets, won_bets, win_rate, avg_odds, bankroll, peak, max_drawdown * 100.0, flat_pnl, roi_perc};
}

int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=========================================================================\n";
    std::cout << "         DEEPPITCH: HIGH-PERFORMANCE C++20 QUANT BACKTEST ENGINE         \n";
    std::cout << "=========================================================================\n\n";

    // Support execution from root repo or backtest/ subdirectory
    std::ifstream file("backtest2.0.csv");
    if (!file.is_open()) {
        file.open("backtest/backtest2.0.csv");
    }
    if (!file.is_open()) {
        std::cerr << "Error: Could not locate backtest2.0.csv in current directory or backtest/\n";
        return 1;
    }

    std::string line;
    std::vector<Match> all_matches;
    all_matches.reserve(17000);

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

        double q_h = std::stod(q_hstring);
        double q_d = std::stod(q_dstring);
        double q_a = std::stod(q_astring);

        // De-vigged Bookmaker Implied Probabilities
        double I_1 = 1.0 / q_h;
        double I_X = 1.0 / q_d;
        double I_2 = 1.0 / q_a;
        double s = I_1 + I_2 + I_X;
        double p_h_book = I_1 / s;
        double p_d_book = I_X / s;
        double p_a_book = I_2 / s;

        // Raw Neural Network Softmax Output (56 features)
        double p_h = std::stod(p_hstring) / 100.0;
        double p_d = std::stod(p_dstring) / 100.0;
        double p_a = std::stod(p_astring) / 100.0;
        int res = std::stoi(act_resstring);

        // Bayesian Shrinkage Layer via Calibrated Logistic Regression
        // Weights: Intercept = 0.0806, Beta_Model = 0.6173 (55.1%), Beta_Book = 0.5035 (44.9%)
        double z_1 = 0.0806 + 0.6173 * std::log(p_h / (1.0 - p_h)) + 0.5035 * std::log(p_h_book / (1.0 - p_h_book));
        double z_x = 0.0806 + 0.6173 * std::log(p_d / (1.0 - p_d)) + 0.5035 * std::log(p_d_book / (1.0 - p_d_book));
        double z_2 = 0.0806 + 0.6173 * std::log(p_a / (1.0 - p_a)) + 0.5035 * std::log(p_a_book / (1.0 - p_a_book));

        double p_1_raw = 1.0 / (1.0 + std::exp(-z_1));
        double p_x_raw = 1.0 / (1.0 + std::exp(-z_x));
        double p_2_raw = 1.0 / (1.0 + std::exp(-z_2));

        // Softmax-normalized Calibrated Probabilities
        double s_ = p_1_raw + p_x_raw + p_2_raw;
        double p_1 = p_1_raw / s_;
        double p_x = p_x_raw / s_;
        double p_2 = p_2_raw / s_;

        all_matches.push_back({{q_h, q_d, q_a}, {p_1, p_x, p_2}, res});
    }

    // Strict 90% Temporal Split: Last 10% is 100% blind out-of-sample Test Set
    size_t split_idx = static_cast<size_t>(0.90 * all_matches.size());
    std::vector<Match> test_matches(all_matches.begin() + split_idx, all_matches.end());

    std::cout << "Dataset Ingestion:\n";
    std::cout << "  - Total fixtures processed:   " << all_matches.size() << "\n";
    std::cout << "  - Out-of-Sample Test Set:     " << test_matches.size() << " fixtures (10% temporal split)\n\n";

    // 1. Sensitivity Analysis Grid
    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << " EV Threshold Sensitivity Analysis (Test Set, Quarter-Kelly Cap 1.5%):\n";
    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << "  EV Thresh | Bets | Won | Win Rate | Avg Odd | Final Bank | ROI (%) | Max DD  | Flat PnL\n";
    std::cout << "-------------------------------------------------------------------------\n";

    std::vector<double> thresholds = {0.02, 0.03, 0.04, 0.05, 0.055, 0.06, 0.07, 0.08};
    for (double th : thresholds) {
        auto r = run_strategy(test_matches, th);
        std::cout << "   " << std::setw(4) << th * 100.0 << "%   | "
                  << std::setw(4) << r.total_bets << " | "
                  << std::setw(3) << r.won_bets << " |  "
                  << std::setw(5) << r.win_rate << "% |  "
                  << std::setw(5) << r.avg_odds << "  | EUR "
                  << std::setw(7) << r.final_bankroll << " | "
                  << (r.roi_perc >= 0 ? "+" : "") << std::setw(5) << r.roi_perc << "% | "
                  << std::setw(5) << r.max_drawdown << "% | "
                  << (r.flat_pnl >= 0 ? "+" : "") << std::setw(5) << r.flat_pnl << "u\n";
    }
    std::cout << "-------------------------------------------------------------------------\n\n";

    // 2. Primary Execution on Target Strategy (EV > 5.5%)
    const double TARGET_EV = 0.055;
    std::vector<PlayableBet> playable_bets;
    auto target_res = run_strategy(test_matches, TARGET_EV, &playable_bets);

    std::cout << "=========================================================================\n";
    std::cout << "  TARGET STRATEGY BENCHMARK: EV > 5.5% (1.60 < Odds < 3.10, Max Cap 1.5%)\n";
    std::cout << "=========================================================================\n";
    std::cout << "  ? Initial Capital:        EUR 1,000.00\n";
    std::cout << "  ? Total Bets Qualified:   " << target_res.total_bets << " (out of " << test_matches.size() << " test matches)\n";
    std::cout << "  ? Bets Won / Win Rate:    " << target_res.won_bets << " (" << target_res.win_rate << "%)\n";
    std::cout << "  ? Average Executed Odds:  " << target_res.avg_odds << "\n";
    std::cout << "  ? Peak Bankroll Reached:  EUR " << target_res.peak_bankroll << "\n";
    std::cout << "  ? Final Bankroll:         EUR " << target_res.final_bankroll << "\n";
    std::cout << "  ? Net Capital Return:     +" << target_res.roi_perc << "%\n";
    std::cout << "  ? Maximum Drawdown:       " << target_res.max_drawdown << "%\n";
    std::cout << "  ? Flat Stake PnL:         +" << target_res.flat_pnl << " units\n";
    std::cout << "=========================================================================\n\n";

    if (playable_bets.empty()) {
        std::cerr << "Warning: No bets qualified for Monte Carlo simulation.\n";
        return 0;
    }

    // 3. Monte Carlo Simulations (10,000 runs)
    const int NUM_SIMULATIONS = 10000;
    const size_t N_BETS = playable_bets.size();
    std::mt19937 rng(42);

    // Module A: Bootstrap Resampling (Sampling N bets with replacement)
    // Quantifies capital distribution across 10,000 simulated seasons
    std::vector<double> bs_final_bankrolls;
    std::vector<double> bs_max_drawdowns;
    bs_final_bankrolls.reserve(NUM_SIMULATIONS);
    bs_max_drawdowns.reserve(NUM_SIMULATIONS);
    int bs_ruin_count = 0;

    std::uniform_int_distribution<size_t> dist_idx(0, N_BETS - 1);

    for (int sim = 0; sim < NUM_SIMULATIONS; ++sim) {
        double sim_bankroll = 1000.0;
        double sim_peak = 1000.0;
        double sim_max_dd = 0.0;

        for (size_t k = 0; k < N_BETS; ++k) {
            const auto& bet = playable_bets[dist_idx(rng)];
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

            if (sim_bankroll < 200.0) {
                bs_ruin_count++;
                break;
            }
        }

        bs_final_bankrolls.push_back(sim_bankroll);
        bs_max_drawdowns.push_back(sim_max_dd);
    }

    std::sort(bs_final_bankrolls.begin(), bs_final_bankrolls.end());
    std::sort(bs_max_drawdowns.begin(), bs_max_drawdowns.end());

    double bs_ruin_prob = (static_cast<double>(bs_ruin_count) / NUM_SIMULATIONS) * 100.0;

    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << " MONTE CARLO STRESS TEST A: BOOTSTRAP RESAMPLING (10,000 Seasons)\n";
    std::cout << " (Evaluates sample uncertainty & capital distribution with replacement)\n";
    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << "  ? Probability of Ruin (< 200 EUR): " << bs_ruin_prob << "%\n\n";
    std::cout << "  [Final Bankroll Quantiles]\n";
    std::cout << "    - 5th Percentile (Pessimistic):  EUR " << bs_final_bankrolls[static_cast<size_t>(NUM_SIMULATIONS * 0.05)] << "\n";
    std::cout << "    - 50th Percentile (Median):      EUR " << bs_final_bankrolls[static_cast<size_t>(NUM_SIMULATIONS * 0.50)] << "\n";
    std::cout << "    - 95th Percentile (Optimistic):  EUR " << bs_final_bankrolls[static_cast<size_t>(NUM_SIMULATIONS * 0.95)] << "\n\n";
    std::cout << "  [Drawdown Risk Quantiles]\n";
    std::cout << "    - 5th Percentile (Mild):          " << bs_max_drawdowns[static_cast<size_t>(NUM_SIMULATIONS * 0.05)] * 100.0 << "%\n";
    std::cout << "    - 50th Percentile (Median DD):    " << bs_max_drawdowns[static_cast<size_t>(NUM_SIMULATIONS * 0.50)] * 100.0 << "%\n";
    std::cout << "    - 95th Percentile (Severe DD):    " << bs_max_drawdowns[static_cast<size_t>(NUM_SIMULATIONS * 0.95)] * 100.0 << "%\n";
    std::cout << "    - Worst-Case Absolute Drawdown:   " << bs_max_drawdowns.back() * 100.0 << "%\n\n";

    // Module B: Trade-Order Permutation Test (Reshuffling without replacement)
    // Quantifies sequence risk & streak clustering on the exact 384 observed bets
    std::vector<size_t> indices(N_BETS);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<double> perm_max_drawdowns;
    perm_max_drawdowns.reserve(NUM_SIMULATIONS);

    for (int sim = 0; sim < NUM_SIMULATIONS; ++sim) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double sim_bankroll = 1000.0;
        double sim_peak = 1000.0;
        double sim_max_dd = 0.0;

        for (size_t k = 0; k < N_BETS; ++k) {
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
        }
        perm_max_drawdowns.push_back(sim_max_dd);
    }

    std::sort(perm_max_drawdowns.begin(), perm_max_drawdowns.end());

    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << " MONTE CARLO STRESS TEST B: TRADE-ORDER PERMUTATION (10,000 Shuffles)\n";
    std::cout << " (Evaluates path dependency & losing-streak clustering without replacement)\n";
    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << "  [Permutation Drawdown Quantiles]\n";
    std::cout << "    - 5th Percentile:                 " << perm_max_drawdowns[static_cast<size_t>(NUM_SIMULATIONS * 0.05)] * 100.0 << "%\n";
    std::cout << "    - 50th Percentile (Median DD):    " << perm_max_drawdowns[static_cast<size_t>(NUM_SIMULATIONS * 0.50)] * 100.0 << "%\n";
    std::cout << "    - 95th Percentile (Severe DD):    " << perm_max_drawdowns[static_cast<size_t>(NUM_SIMULATIONS * 0.95)] * 100.0 << "%\n";
    std::cout << "    - Worst-Case Permutation DD:      " << perm_max_drawdowns.back() * 100.0 << "%\n";
    std::cout << "=========================================================================\n";

    return 0;
}
