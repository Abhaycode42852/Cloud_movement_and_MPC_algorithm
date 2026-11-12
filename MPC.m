function [S, g1_dbg, g0_dbg, iL1_pred_dbg, ...
          iL0_pred_dbg, S_prev_dbg, Jmin_dbg, ...
          deltaJ_dbg, switch_event_dbg, e_iL_dbg] = ...
          MPC_Boost(iL, Vin, Vout, iL_ref)

Ts = 15e-6;
L = 0.00166914;
RL = 0;
wf = 0.001;

States = [1; 0];
g_op = inf;
n_op = 1;

persistent S_old;
if isempty(S_old)
    S_old = 0;
end

% Record previous state before making the new decision
S_prev_dbg = S_old;

% Initialize diagnostic outputs
g1_dbg = 0;
g0_dbg = 0;
iL1_pred_dbg = 0;
iL0_pred_dbg = 0;

for n = 1:2

    % Predict inductor current for each candidate switching state
    iL_k1 = iL + (Ts/L) * ...
        (Vin - iL*RL - Vout*(1 - States(n)));

    % Switching penalty
    g_sw = abs(States(n) - S_old);

    % Current-tracking error
    g_iL = abs(iL_ref - iL_k1);

    % Total cost
    g = g_iL^2 + wf*(g_sw^2);

    % Diagnostic outputs
    if n == 1
        iL1_pred_dbg = iL_k1;
        g1_dbg = g;
    else
        iL0_pred_dbg = iL_k1;
        g0_dbg = g;
    end

    % Original decision logic — unchanged
    if g < g_op
        n_op = n;
        g_op = g;
    end
end


S = States(n_op);
Jmin_dbg = g_op;

% Diagnostic outputs
deltaJ_dbg = g1_dbg - g0_dbg;
switch_event_dbg = double(S ~= S_prev_dbg);
e_iL_dbg = iL_ref - iL;

% Update stored state after recording the previous state
S_old = S;


end