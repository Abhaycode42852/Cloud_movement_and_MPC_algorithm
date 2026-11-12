function [ipv_ref, Pexp, fitValid, ipv_c1_dbg, ipv_c2_dbg, ...
          P1_dbg, P2_dbg, DeltaP_dbg, DeltaP_filt_dbg, Iref_max, i2_dbg, ...
          minGap, currentSpan, fitStatus] = ...
          Ref_Gen(ipv_meas, vpv_meas, G_ft, enable)
% Ref_Gen: irradiance-normalized curve fitting with future-Irradiance MPPT.
%
% PURPOSE
%   Use each measured PV point and its associated irradiance to translate
%   all historical samples to a common reference irradiance. Fit the common
%   V-I curve by least squares, translate that fitted curve to G_ft (treated
%   as G(k+1)), and choose I+ or I- according to predicted future power.
%
% INPUTS
%   ipv_meas : measured PV current (A)
%   vpv_meas : measured PV voltage (V)
%   G_ft     : available future irradiance, nominally G(k+1) (W/m^2)
%   enable   : retained for interface compatibility; selection is external
%
% OUTPUTS
%   ipv_ref          : predicted future PV-current / inductor-current ref (A)
%   Pexp             : model-predicted power at ipv_ref (W)
%   fitValid         : 1 when a validated normalized model is available
%   ipv_c1_dbg       : future candidate I+ (A)
%   ipv_c2_dbg       : future candidate I- (A)
%   P1_dbg           : predicted future power at I+ (W)
%   P2_dbg           : predicted future power at I- (W)
%   DeltaP_dbg       : P(I+) - P(I-) (W)
%   DeltaP_filt_dbg  : filtered candidate-power difference (W)
%   Iref_max         : upper future PV-current reference limit (A)
%   i2_dbg           : measured PV current (A)
%   minGap           : minimum separation of stored normalized currents (A)
%   currentSpan      : span of stored normalized currents (A)
%   fitStatus        : diagnostic code (listed below)
%
% NORMALIZATION AND FUTURE-IRRADIANCE TRANSLATION
%   G_REF = 1000 W/m^2. Each measurement is paired with Gprev because the
%   PV irradiance input is assumed to have one MPPT-period (100 us) delay.
%   For a sample measured at Gs:
%       Inorm = Imeas * G_REF/Gs
%       Vnorm = Vmeas + A_G*log(G_REF/Gs) - Rs_eff*(Inorm - Imeas)
%   For a future candidate I at G_ft, r = G_ft/G_REF:
%       Ibase = I/r
%       Vfuture(I) = Vfit(Ibase) + A_G*log(r) - Rs_eff*(I - Ibase)
%
%   A_G and Rs_eff below are provisional estimates from plotted I-V curves.
%   They must be validated against numerical TerraSAS/PV Array data. The
%   translation is an approximation; it assumes temperature is constant.
%
% CANDIDATE SEARCH
%   A current reference is initialized from the maximum of the translated
%   fitted curve. Subsequently, candidate currents I+/- = center +/- DeltaI
%   are evaluated using the translated curve. A filtered power difference
%   selects I+ or I-; a deadband prevents needless reference chatter.
%
% fitStatus:
%   0  = invalid/too-low future irradiance
%   2  = invalid PV measurement
%   3  = fewer than MIN_HISTORY accepted samples
%   4  = invalid normalized sample/model data
%   5  = insufficient current separation or span
%   6  = ill-conditioned least-squares fit
%   7  = invalid fitted coefficients
%   8  = future candidate/root voltage unsupported or invalid
%   9  = invalid predicted power
%   10 = normalized curve fit and future candidate evaluation valid
%   11 = no acceptable MPP inside the measured current support
%   12 = fitted-curve residual exceeds allowed threshold

% ------------------------- Constants ------------------------------
G_REF = 1000.0;             % W/m^2, common normalization irradiance
Isc_STC = 5.1;              % A, PV module reference Isc
Voc_STC = 31.0;             % V, PV module reference Voc
G_MIN = 20.0;               % W/m^2, below this forecast is invalid

% Provisional curve-translation parameters. Validate using numerical I-V data.
A_G = 1.35;                 % V, logarithmic irradiance voltage coefficient
Rs_eff = 0.50;              % ohm, effective module series-resistance correction

Ts_MPPT = 100e-6;           % Expected function-call interval
T_COLLECT_START = 0.10;     % Avoid earliest converter startup transient (s)
collectStartCalls = max(1, round(T_COLLECT_START/Ts_MPPT)); %1000

MAX_HISTORY = 7;
MIN_HISTORY = 5;
historySeparation = 0.015;  % A at G_REF; new point must be this distinct
minPairGap = 0.010;         % A, minimum pair separation
minSpan = 0.040;            % A, minimum normalized-current span
minGramDet = 1e-8;          % conditioning check for normalized least squares

SETTLE_WINDOW_CALLS = 10;   % 1 ms at a 100 us function period
stableWindowIRange = 0.20;  % A, maximum measured-current range in window
stableWindowVRange = 0.75;  % V, maximum measured-voltage range in window

voltageEnvelopeTol = 0.75;  % V tolerance outside fitted sample envelope
maxFitRmsError = 1.00;      % V RMS residual limit (translation is approximate)
maxFitAbsError = 2.00;      % V absolute residual limit
supportTolNorm = 0.04;      % A allowance for +/- candidates near sample boundary

GjumpFrac = 0.10;            % only resets the short settling window, not history
FIT_CONFIRM_CALLS = 5;       % consecutive valid evaluations needed to select model
FIT_FAIL_GRACE_CALLS = 5;    % retain last model for brief fitting failures
DeltaI = 0.020;              % A, candidate spacing in future-current domain
alphaDP = 0.20;              % low-pass coefficient for P(I+) - P(I-)
P_deadband = 0.05;            % W, candidate decision deadband
GRID_N = 101;                % fixed grid for initial MPP search (codegen friendly)

% ------------------------- Output defaults ------------------------
ipv_ref = 0.0;
Pexp = 0.0;
fitValid = 0.0;
ipv_c1_dbg = 0.0;
ipv_c2_dbg = 0.0;
P1_dbg = 0.0;
P2_dbg = 0.0;
DeltaP_dbg = 0.0;
DeltaP_filt_dbg = 0.0;
Iref_max = 0.0;
i2_dbg = 0.0;
minGap = 0.0;
currentSpan = 0.0;
fitStatus = 0.0;

% ------------------------- Persistent state -----------------------
persistent InormBuf VnormBuf nHist
persistent Gprev GprevValid callCount initialized
persistent recentI recentV recentCount recentIndex
persistent modelValid cLast ImeanLast IscaleLast
persistent IminLast ImaxLast VminLast VmaxLast
persistent lastRefNorm lastRefNormValid DeltaP_filt
persistent fitGoodCount fitBadCount fitReady

if isempty(initialized)
    InormBuf = zeros(1, MAX_HISTORY);
    VnormBuf = zeros(1, MAX_HISTORY);
    nHist = 0;

    Gprev = 0.0;
    GprevValid = false;
    callCount = 0;

    recentI = zeros(1, SETTLE_WINDOW_CALLS);
    recentV = zeros(1, SETTLE_WINDOW_CALLS);
    recentCount = 0;
    recentIndex = 0;

    modelValid = false;
    cLast = zeros(3,1);
    ImeanLast = 0.0;
    IscaleLast = 1.0;
    IminLast = 0.0;
    ImaxLast = 0.0;
    VminLast = 0.0;
    VmaxLast = 0.0;

    lastRefNorm = 0.0;
    lastRefNormValid = false;
    DeltaP_filt = 0.0;
    fitGoodCount = 0;
    fitBadCount = 0;
    fitReady = false;
    initialized = true;
end
callCount = callCount + 1;

% ------------------------- Forecast validation --------------------
forecastValid = ~isnan(G_ft) && ~isinf(G_ft) && G_ft >= G_MIN;
if ~forecastValid
    InormBuf = zeros(1, MAX_HISTORY);
    VnormBuf = zeros(1, MAX_HISTORY);
    nHist = 0;
    Gprev = 0.0;
    GprevValid = false;
    callCount = 0;
    recentI = zeros(1, SETTLE_WINDOW_CALLS);
    recentV = zeros(1, SETTLE_WINDOW_CALLS);
    recentCount = 0;
    recentIndex = 0;
    modelValid = false;
    cLast = zeros(3,1);
    ImeanLast = 0.0;
    IscaleLast = 1.0;
    IminLast = 0.0;
    ImaxLast = 0.0;
    VminLast = 0.0;
    VmaxLast = 0.0;
    lastRefNorm = 0.0;
    lastRefNormValid = false;
    DeltaP_filt = 0.0;
    fitGoodCount = 0;
    fitBadCount = 0;
    fitReady = false;
    fitStatus = 0;
    return;
end

Isc_target = Isc_STC * G_ft / G_REF;
Iref_min = max(0.02 * Isc_target, 0.001);
Iref_max = max(0.97 * Isc_target, Iref_min + 1e-4);
Iref_max = min(Iref_max, 0.999 * Isc_STC);
Iref_max = max(Iref_max, Iref_min + 1e-4);
Iref_max = min(Iref_max, Isc_STC);

% `enable` is retained for port compatibility; the external selector uses
% fitValid. Sampling is allowed while P&O is selected.

i2_dbg = ipv_meas;

% ------------------------- Detect forecast step -------------------
stepDetected = false;
if GprevValid && Gprev >= G_MIN
    stepDetected = abs(G_ft - Gprev) >= GjumpFrac * max(Gprev, G_MIN);
end

% The PV sample at this call corresponds to the previous forecast Gprev.
measurementValid = ~isnan(ipv_meas) && ~isinf(ipv_meas) && ipv_meas > 0 && ...
                   ~isnan(vpv_meas) && ~isinf(vpv_meas) && vpv_meas > 0;

% ------------------------- Update settling window -----------------
measurementSettled = false;
if stepDetected
    % A new target curve has a different candidate-power difference. Reset
    % its filter so the previous irradiance cannot bias the next decision.
    DeltaP_filt = 0.0;
    recentI = zeros(1, SETTLE_WINDOW_CALLS);
    recentV = zeros(1, SETTLE_WINDOW_CALLS);
    recentCount = 0;
    recentIndex = 0;
end

if measurementValid
    recentIndex = recentIndex + 1;
    if recentIndex > SETTLE_WINDOW_CALLS
        recentIndex = 1;
    end
    recentI(recentIndex) = ipv_meas;
    recentV(recentIndex) = vpv_meas;
    recentCount = min(recentCount + 1, SETTLE_WINDOW_CALLS);

    if recentCount >= SETTLE_WINDOW_CALLS
        recentIRange = max(recentI) - min(recentI);
        recentVRange = max(recentV) - min(recentV);
        measurementSettled = recentIRange <= stableWindowIRange && ...
                             recentVRange <= stableWindowVRange;
    end
end

% ------------------------- Normalize and store sample -------------
% Do not clear curve history at irradiance steps: each stored point is
% transformed to G_REF using its own associated irradiance Gprev.
if measurementValid && callCount >= collectStartCalls && ...
        measurementSettled && GprevValid && Gprev >= G_MIN

    Gsample = Gprev;
    normRatio = G_REF / Gsample;
    newInorm = ipv_meas * normRatio;
    newVnorm = vpv_meas + A_G * log(normRatio) ...
               - Rs_eff * (newInorm - ipv_meas);

    normalizedSampleValid = ~isnan(newInorm) && ~isinf(newInorm) && ...
        ~isnan(newVnorm) && ~isinf(newVnorm) && newInorm > 0 && ...
        newVnorm > 0 && newVnorm <= 1.5 * Voc_STC;

    if normalizedSampleValid
        if nHist == 0
            nHist = 1;
            InormBuf(1) = newInorm;
            VnormBuf(1) = newVnorm;
        else
            novelSample = true;
            closestIndex = 1;
            closestDistance = inf;
            for k = 1:nHist
                dist = abs(newInorm - InormBuf(k));
                if dist < closestDistance
                    closestDistance = dist;
                    closestIndex = k;
                end
                if dist < historySeparation
                    novelSample = false;
                end
            end

            if novelSample && nHist < MAX_HISTORY
                nHist = nHist + 1;
                InormBuf(nHist) = newInorm;
                VnormBuf(nHist) = newVnorm;
            elseif novelSample && nHist >= MAX_HISTORY
                InormBuf(closestIndex) = newInorm;
                VnormBuf(closestIndex) = newVnorm;
            end
        end
    end
end

% ------------------------- Fit normalized V-I curve ---------------
fitSucceeded = false;
fitStatus = 3;
fitImean = 0.0;
fitIscale = 1.0;
cFit = zeros(3,1);
fitImin = 0.0;
fitImax = 0.0;
fitVmin = 0.0;
fitVmax = 0.0;

if ~measurementValid
    fitStatus = 2;
elseif nHist < MIN_HISTORY
    fitStatus = 3;
else
    % Current diversity and range checks on normalized samples.
    fitImin = InormBuf(1);
    fitImax = InormBuf(1);
    fitVmin = VnormBuf(1);
    fitVmax = VnormBuf(1);
    minGap = inf;
    for i = 1:nHist
        fitImin = min(fitImin, InormBuf(i));
        fitImax = max(fitImax, InormBuf(i));
        fitVmin = min(fitVmin, VnormBuf(i));
        fitVmax = max(fitVmax, VnormBuf(i));
        for j = i+1:nHist
            minGap = min(minGap, abs(InormBuf(i) - InormBuf(j)));
        end
    end
    currentSpan = fitImax - fitImin;

    if minGap < minPairGap || currentSpan < minSpan
        fitStatus = 5;
    else
        fitImean = 0.0;
        for i = 1:nHist
            fitImean = fitImean + InormBuf(i);
        end
        fitImean = fitImean / nHist;

        fitIscale = 0.0;
        for i = 1:nHist
            fitIscale = max(fitIscale, abs(InormBuf(i) - fitImean));
        end

        if fitIscale <= 1e-6
            fitStatus = 6;
        else
            % Fit V = c(1)*x^2 + c(2)*x + c(3), x=(I-Imean)/Iscale,
            % using the fixed-size 3-by-3 normal equations (codegen friendly).
            s1 = 0.0; s2 = 0.0; s3 = 0.0; s4 = 0.0;
            t0 = 0.0; t1 = 0.0; t2 = 0.0;
            for i = 1:nHist
                xi = (InormBuf(i) - fitImean) / fitIscale;
                vi = VnormBuf(i);
                x2 = xi * xi;
                s1 = s1 + xi;
                s2 = s2 + x2;
                s3 = s3 + x2 * xi;
                s4 = s4 + x2 * x2;
                t0 = t0 + vi;
                t1 = t1 + xi * vi;
                t2 = t2 + x2 * vi;
            end

            M = [s4, s3, s2; s3, s2, s1; s2, s1, double(nHist)];
            rhs = [t2; t1; t0];
            detM = det(M);

            if isnan(detM) || isinf(detM) || abs(detM) < minGramDet
                fitStatus = 6;
            else
                cTry = M \ rhs;
                coeffOK = all(~isnan(cTry)) && all(~isinf(cTry));
                if ~coeffOK
                    fitStatus = 7;
                else
                    sumErr2 = 0.0;
                    maxAbsErr = 0.0;
                    for i = 1:nHist
                        xi = (InormBuf(i) - fitImean) / fitIscale;
                        vhat = cTry(1)*xi*xi + cTry(2)*xi + cTry(3);
                        err = vhat - VnormBuf(i);
                        sumErr2 = sumErr2 + err*err;
                        maxAbsErr = max(maxAbsErr, abs(err));
                    end
                    rmsErr = sqrt(sumErr2 / nHist);
                    if rmsErr > maxFitRmsError || maxAbsErr > maxFitAbsError
                        fitStatus = 12;
                    else
                        cFit = cTry;
                        fitSucceeded = true;
                        fitStatus = 10;
                    end
                end
            end
        end
    end
end

% ------------------------- Store the last validated model ---------
if fitSucceeded
    cLast = cFit;
    ImeanLast = fitImean;
    IscaleLast = fitIscale;
    IminLast = fitImin;
    ImaxLast = fitImax;
    VminLast = fitVmin;
    VmaxLast = fitVmax;
    modelValid = true;

    fitGoodCount = min(fitGoodCount + 1, FIT_CONFIRM_CALLS);
    fitBadCount = 0;
    if fitGoodCount >= FIT_CONFIRM_CALLS
        fitReady = true;
    end
else
    fitGoodCount = 0;
    if fitReady
        fitBadCount = fitBadCount + 1;
        if fitBadCount >= FIT_FAIL_GRACE_CALLS
            fitReady = false;
            fitBadCount = 0;
        end
    else
        fitBadCount = 0;
    end
end

% ------------------------- Translate future curve and choose I+/I- -
% This section uses the most recent validated fit, even through brief fit
% failures. This avoids dropping the feedforward reference for one bad call.
futureReferenceValid = false;
if modelValid
    rFuture = G_ft / G_REF;
    logR = log(rFuture);

    % On first available model, initialize the reference using a fixed-grid
    % maximum search on the translated future curve, restricted to measured
    % current support. No extrapolated root is accepted as the anchor.
    if ~lastRefNormValid
        supportLow = max(Iref_min, rFuture * IminLast);
        supportHigh = min(Iref_max, rFuture * ImaxLast);
        bestP = -inf;
        bestI = 0.0;
        bestGridIndex = 0;

        if supportHigh > supportLow
            for gi = 0:(GRID_N-1)
                frac = double(gi) / double(GRID_N-1);
                Itry = supportLow + frac * (supportHigh - supportLow);
                Ibase = Itry / rFuture;
                xtry = (Ibase - ImeanLast) / IscaleLast;
                Vbase = cLast(1)*xtry*xtry + cLast(2)*xtry + cLast(3);
                Vtry = Vbase + A_G*logR - Rs_eff*(Itry - Ibase);
                if Ibase >= IminLast && Ibase <= ImaxLast && ...
                        Vbase >= VminLast - voltageEnvelopeTol && ...
                        Vbase <= VmaxLast + voltageEnvelopeTol && ...
                        Vtry > 0 && Vtry <= 1.2*Voc_STC
                    Ptry = Itry * Vtry;
                    if ~isnan(Ptry) && ~isinf(Ptry) && Ptry > bestP
                        bestP = Ptry;
                        bestI = Itry;
                        bestGridIndex = gi;
                    end
                end
            end
        end

        if bestGridIndex <= 1 || bestGridIndex >= (GRID_N-1) || bestP <= 0
            modelValid = false;
            fitReady = false;
            fitGoodCount = 0;
            fitStatus = 11;
        else
            lastRefNorm = bestI / rFuture;
            lastRefNormValid = true;
            DeltaP_filt = 0.0;
        end
    end

    if modelValid && lastRefNormValid
        % Keep both candidate points inside the translated measured-current
        % support. Reduce DeltaI when the available sample span is narrow.
        targetSpan = rFuture * (ImaxLast - IminLast);
        DeltaILocal = min(DeltaI, 0.45 * targetSpan);
        centerLow = max(Iref_min, rFuture*IminLast + DeltaILocal);
        centerHigh = min(Iref_max, rFuture*ImaxLast - DeltaILocal);

        if centerHigh <= centerLow || DeltaILocal <= 1e-6
            fitStatus = 8;
        else
            center = lastRefNorm * rFuture;
            center = min(max(center, centerLow), centerHigh);
            Iplus = min(max(center + DeltaILocal, Iref_min), Iref_max);
            Iminus = min(max(center - DeltaILocal, Iref_min), Iref_max);

            IplusBase = Iplus / rFuture;
        IminusBase = Iminus / rFuture;
        xPlus = (IplusBase - ImeanLast) / IscaleLast;
        xMinus = (IminusBase - ImeanLast) / IscaleLast;
        VplusBase = cLast(1)*xPlus*xPlus + cLast(2)*xPlus + cLast(3);
        VminusBase = cLast(1)*xMinus*xMinus + cLast(2)*xMinus + cLast(3);
        Vplus = VplusBase + A_G*logR - Rs_eff*(Iplus - IplusBase);
        Vminus = VminusBase + A_G*logR - Rs_eff*(Iminus - IminusBase);

        plusSupported = IplusBase >= IminLast-supportTolNorm && ...
            IplusBase <= ImaxLast+supportTolNorm && ...
            VplusBase >= VminLast-voltageEnvelopeTol && ...
            VplusBase <= VmaxLast+voltageEnvelopeTol && ...
            Vplus > 0 && Vplus <= 1.2*Voc_STC;
        minusSupported = IminusBase >= IminLast-supportTolNorm && ...
            IminusBase <= ImaxLast+supportTolNorm && ...
            VminusBase >= VminLast-voltageEnvelopeTol && ...
            VminusBase <= VmaxLast+voltageEnvelopeTol && ...
            Vminus > 0 && Vminus <= 1.2*Voc_STC;

        ipv_c1_dbg = Iplus;
        ipv_c2_dbg = Iminus;
        if plusSupported
            Pplus = Iplus * Vplus;
        else
            Pplus = -1.0;
        end
        if minusSupported
            Pminus = Iminus * Vminus;
        else
            Pminus = -1.0;
        end

        if Pplus > 0 && Pminus > 0 && ...
                ~isnan(Pplus) && ~isinf(Pplus) && ...
                ~isnan(Pminus) && ~isinf(Pminus)
            P1_dbg = Pplus;
            P2_dbg = Pminus;
            DeltaP_dbg = Pplus - Pminus;
            if stepDetected
                % Use the newly translated curve immediately at a previewed step.
                DeltaP_filt = DeltaP_dbg;
            else
                DeltaP_filt = (1-alphaDP)*DeltaP_filt + alphaDP*DeltaP_dbg;
            end
            DeltaP_filt_dbg = DeltaP_filt;

            if DeltaP_filt > P_deadband
                ipv_ref = Iplus;
            elseif DeltaP_filt < -P_deadband
                ipv_ref = Iminus;
            else
                ipv_ref = center;
            end
            Pexp = ipv_ref * ( ...
                (cLast(1)*((ipv_ref/rFuture-ImeanLast)/IscaleLast)^2) + ...
                (cLast(2)*((ipv_ref/rFuture-ImeanLast)/IscaleLast)) + ...
                 cLast(3) + A_G*logR - Rs_eff*(ipv_ref - ipv_ref/rFuture));

            if Pexp <= 0 || isnan(Pexp) || isinf(Pexp)
                Pexp = 0.0;
                fitStatus = 9;
            elseif fitSucceeded
                fitStatus = 10;
            end

            % Store the chosen reference in common-irradiance coordinates.
            lastRefNorm = ipv_ref / rFuture;
            lastRefNormValid = true;
            futureReferenceValid = true;
            else
                fitStatus = 8;
                Pexp = 0.0;
                ipv_ref = center;
            end
        end
    end
end

% ------------------------- Safe fallback / outputs ----------------
if ~modelValid || ~lastRefNormValid
    if lastRefNormValid
        ipv_ref = min(max(lastRefNorm * (G_ft/G_REF), Iref_min), Iref_max);
    elseif modelValid
        ipv_ref = min(max(0.80*Isc_target, Iref_min), Iref_max);
    else
        ipv_ref = min(max(0.80*Isc_target, Iref_min), Iref_max);
    end
    Pexp = 0.0;
end

fitValid = double(fitReady && modelValid && lastRefNormValid && futureReferenceValid);

% Save current forecast for pairing with the next delayed PV sample.
Gprev = G_ft;
GprevValid = true;

end