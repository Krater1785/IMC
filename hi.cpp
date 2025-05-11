#include<bits/stdc++.h>
using namespace std;

double arr[4][4] = {
    {1, 1.45, 0.52, 0.72},
    {0.7, 1, 0.31, 0.48},
    {1.95, 3.1, 1, 1.49},
    {1.34, 1.98, 0.64, 1}
};

double dp[6][5];
int parent[6][5];  // To trace the path

// Recursive DP function with memoization
double rec(int step, int curr, int dest) {
    if (step == 1) {
        return arr[curr][dest];
    }

    if (dp[step][curr] > -500.0) return dp[step][curr];

    double maxProfit = -1e9;
    for (int i = 0; i < 4; i++) {
        double profit = arr[curr][i] * rec(step - 1, i, dest);
        if (profit > maxProfit) {
            maxProfit = profit;
            parent[step][curr] = i;
        }
    }
    return dp[step][curr] = maxProfit;
}

// Reconstruct the path from the parent table
void printPath(int step, int curr, int dest) {
    vector<int> path;
    path.push_back(curr);
    for (int s = step; s > 1; --s) {
        curr = parent[s][curr];
        path.push_back(curr);
    }
    path.push_back(dest);

    for (int i = 0; i < path.size(); ++i) {
        cout << path[i];
        if (i != path.size() - 1) cout << " -> ";
    }
    cout << endl;
}

int main() {
    int steps = 4; // Total steps: 5 trades
    int start = 3, end = 3;

    // Initialize dp and parent arrays
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 5; ++j) {
            dp[i][j] = -1000.0;
            parent[i][j] = -1;
        }

    double maxPnl = rec(steps, start, end);
    cout << "Best path for 5 trades starting and ending in currency 3:\n";
    printPath(steps, start, end);
    cout << "Max PnL: " << maxPnl << endl;

    return 0;
}
