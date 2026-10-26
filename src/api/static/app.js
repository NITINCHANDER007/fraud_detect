// AegisGuard Risk Engine - Interactive Client Dashboard Controller

document.addEventListener('DOMContentLoaded', async () => {
  // Primary elements
  const scoreForm = document.getElementById('score-form');
  const amountInput = document.getElementById('amount');
  const txIdInput = document.getElementById('tx_id');
  const btnScore = document.getElementById('btn-score');
  const btnBatchDemo = document.getElementById('btn-batch-demo');
  const presetButtons = document.querySelectorAll('.preset-btn');
  const remainingDrawer = document.getElementById('remaining-features');
  
  // Output elements
  const gaugeFill = document.getElementById('gauge-fill');
  const gaugeScore = document.getElementById('gauge-score');
  const liveLatency = document.getElementById('live-latency');
  const decisionBadge = document.getElementById('decision-badge');
  const decisionIcon = document.getElementById('decision-icon');
  const decisionText = document.getElementById('decision-text');
  const metaTier = document.getElementById('meta-tier');
  const metaThreshold = document.getElementById('meta-threshold');
  const metaAction = document.getElementById('meta-action');
  const factorsList = document.getElementById('risk-factors-list');
  const thresholdSlider = document.getElementById('threshold-slider');
  const sliderThresholdVal = document.getElementById('slider-threshold-val');
  const batchSection = document.getElementById('batch-section');
  const batchTableBody = document.getElementById('batch-table-body');
  const batchSummaryChips = document.getElementById('batch-summary-chips');

  // State cache
  let currentOptimalThreshold = 0.7466;
  let lastRiskScore = 0.0;
  let presetData = {};

  // 1. Populate all V1-V28 inputs in the drawer
  const keyFeatures = ['V14', 'V10', 'V17', 'V12', 'V4', 'V11'];
  for (let i = 1; i <= 28; i++) {
    const fName = `V${i}`;
    if (!keyFeatures.includes(fName)) {
      const group = document.createElement('div');
      group.className = 'feature-input-group';
      group.innerHTML = `
        <label for="v${i}">${fName}</label>
        <input type="number" step="0.001" id="v${i}" value="0.000" class="form-input mono">
      `;
      remainingDrawer.appendChild(group);
    }
  }

  // 2. Fetch Sample Presets from API
  try {
    const res = await fetch('/v1/sample-transactions');
    if (res.ok) {
      const samples = await res.json();
      presetData = {
        'legit-coffee': samples[0].payload,
        'legit-high': samples[1].payload,
        'fraud-bot': samples[2].payload,
        'fraud-ato': samples[3].payload
      };
    }
  } catch (err) {
    console.warn('Using offline presets fallback:', err);
  }

  // 3. Preset Switcher
  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      presetButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const presetKey = btn.dataset.preset;
      const data = presetData[presetKey];
      if (data) {
        applyPreset(data);
        scoreTransaction();
      }
    });
  });

  function applyPreset(data) {
    txIdInput.value = data.transaction_id || `tx_${Math.random().toString(36).substring(2, 9)}`;
    amountInput.value = data.Amount.toFixed(2);

    for (let i = 1; i <= 28; i++) {
      const el = document.getElementById(`v${i}`);
      if (el && data[`V${i}`] !== undefined) {
        el.value = data[`V${i}`].toFixed(4);
      }
    }
  }

  // 4. Form Submit & Scoring
  scoreForm.addEventListener('submit', (e) => {
    e.preventDefault();
    scoreTransaction();
  });

  async function scoreTransaction() {
    btnScore.disabled = true;
    btnScore.innerHTML = `<span>Scoring...</span>`;

    const payload = {
      transaction_id: txIdInput.value || 'tx_manual',
      Amount: parseFloat(amountInput.value) || 0.0
    };

    for (let i = 1; i <= 28; i++) {
      const el = document.getElementById(`v${i}`);
      payload[`V${i}`] = el ? parseFloat(el.value) || 0.0 : 0.0;
    }

    try {
      const start = performance.now();
      const res = await fetch('/v1/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const clientLatency = (performance.now() - start).toFixed(1);

      if (!res.ok) throw new Error(`API error: ${res.statusText}`);
      const data = await res.json();

      updateDecisionUI(data, clientLatency);
    } catch (err) {
      console.error(err);
      alert('Error scoring transaction: ' + err.message);
    } finally {
      btnScore.disabled = false;
      btnScore.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>
        Score Transaction in Real-Time
      `;
    }
  }

  function updateDecisionUI(data, clientLatency) {
    lastRiskScore = data.risk_score;
    currentOptimalThreshold = data.decision_threshold;
    thresholdSlider.value = currentOptimalThreshold;
    sliderThresholdVal.innerText = currentOptimalThreshold.toFixed(4);

    // Update Latency
    liveLatency.innerText = `Server: ${data.latency_ms.toFixed(2)} ms (E2E: ${clientLatency} ms)`;

    // Update Circular Radial Gauge
    // Circumference = 2 * PI * 65 ≈ 408.4
    const circumference = 408.4;
    const offset = circumference - (data.risk_score * circumference);
    gaugeFill.style.strokeDashoffset = offset;
    gaugeScore.innerText = `${(data.risk_score * 100).toFixed(1)}%`;

    // Dynamic Color Shift
    decisionBadge.className = 'decision-badge';
    if (data.risk_score < 0.20) {
      gaugeFill.style.stroke = '#10B981';
      decisionBadge.classList.add('approved');
      decisionIcon.innerText = '✓';
      decisionText.innerText = 'APPROVE';
      metaAction.innerText = 'Immediate Zero-Friction Authorization';
    } else if (data.risk_score < currentOptimalThreshold) {
      gaugeFill.style.stroke = '#F59E0B';
      decisionBadge.classList.add('review');
      decisionIcon.innerText = '⚠';
      decisionText.innerText = 'MANUAL REVIEW';
      metaAction.innerText = 'Trigger Step-Up 2FA / OTP Verification';
    } else {
      gaugeFill.style.stroke = '#EF4444';
      decisionBadge.classList.add('declined');
      decisionIcon.innerText = '✕';
      decisionText.innerText = 'DECLINE FRAUD';
      metaAction.innerText = 'Block Charge & Alert Cardholder';
    }

    metaTier.innerText = data.risk_tier;
    metaThreshold.innerText = data.decision_threshold.toFixed(4);

    // Update Top Risk Contributors
    if (data.top_risk_factors && data.top_risk_factors.length > 0) {
      const maxContrib = Math.max(...data.top_risk_factors.map(f => f.contribution), 0.01);
      factorsList.innerHTML = data.top_risk_factors.map(f => {
        const pct = Math.min((f.contribution / maxContrib) * 100, 100);
        return `
          <div class="factor-item">
            <div class="factor-header">
              <span><strong>${f.feature}</strong> (val: ${f.feature_value})</span>
              <span>weight: ${f.contribution.toFixed(3)}</span>
            </div>
            <div class="factor-bar-bg">
              <div class="factor-bar-fill" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
      }).join('');
    } else {
      factorsList.innerHTML = `<div class="factor-placeholder">Normal transaction profile (low anomaly weights)</div>`;
    }
  }

  // 5. Interactive Threshold Slider
  thresholdSlider.addEventListener('input', () => {
    const th = parseFloat(thresholdSlider.value);
    sliderThresholdVal.innerText = th.toFixed(4);
    metaThreshold.innerText = th.toFixed(4);

    // Re-evaluate current score
    if (lastRiskScore < 0.20) {
      decisionBadge.className = 'decision-badge approved';
      decisionText.innerText = 'APPROVE';
      metaTier.innerText = 'LOW';
    } else if (lastRiskScore < th) {
      decisionBadge.className = 'decision-badge review';
      decisionText.innerText = 'MANUAL REVIEW';
      metaTier.innerText = 'MEDIUM';
    } else {
      decisionBadge.className = 'decision-badge declined';
      decisionText.innerText = 'DECLINE FRAUD';
      metaTier.innerText = 'HIGH';
    }
  });

  // 6. Batch Demo Execution
  btnBatchDemo.addEventListener('click', async () => {
    btnBatchDemo.disabled = true;
    btnBatchDemo.innerText = 'Scoring 50 Transactions...';
    batchSection.style.display = 'block';

    // Generate batch with realistic variance (45 legit, 5 fraud bursts)
    const batch = [];
    for (let i = 0; i < 50; i++) {
      const isFraudSample = i >= 45;
      const basePreset = isFraudSample 
        ? presetData['fraud-bot'] || presetData['fraud-ato']
        : presetData['legit-coffee'] || presetData['legit-high'];

      const tx = {
        transaction_id: `tx_batch_${1000 + i}`,
        Amount: basePreset.Amount + (Math.random() * 20 - 10)
      };
      for (let v = 1; v <= 28; v++) {
        tx[`V${v}`] = basePreset[`V${v}`] + (Math.random() * 0.2 - 0.1);
      }
      batch.push(tx);
    }

    try {
      const res = await fetch('/v1/batch-score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transactions: batch })
      });
      const data = await res.json();

      batchSummaryChips.innerHTML = `
        <span class="pill-badge" style="background: rgba(16,185,129,0.15); color: #10B981">
          ✓ Approved: ${data.approved_count}
        </span>
        <span class="pill-badge" style="background: rgba(245,158,11,0.15); color: #F59E0B">
          ⚠ Review: ${data.review_required_count}
        </span>
        <span class="pill-badge" style="background: rgba(239,68,68,0.15); color: #EF4444">
          ✕ Blocked: ${data.flagged_fraud_count}
        </span>
        <span class="pill-badge" style="background: rgba(6,182,212,0.15); color: #06B6D4">
          ⚡ Latency: ${data.total_latency_ms} ms (${data.average_latency_per_tx_ms} ms/tx)
        </span>
      `;

      batchTableBody.innerHTML = data.results.slice(0, 15).map(r => `
        <tr>
          <td><code>${r.transaction_id}</code></td>
          <td>$${(Math.random() * 80 + 10).toFixed(2)}</td>
          <td><strong>${(r.risk_score * 100).toFixed(1)}%</strong></td>
          <td>${r.risk_tier}</td>
          <td>
            <span class="pill-badge" style="background: ${r.is_fraud ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)'}; color: ${r.is_fraud ? '#EF4444' : '#10B981'}">
              ${r.recommended_action}
            </span>
          </td>
          <td><code>V14, V10, V17</code></td>
        </tr>
      `).join('');
    } catch (err) {
      console.error(err);
    } finally {
      btnBatchDemo.disabled = false;
      btnBatchDemo.innerText = '⚡ Run 50-Tx Batch Benchmark';
    }
  });

  // Automatically score initial preset
  if (presetData['legit-coffee']) {
    applyPreset(presetData['legit-coffee']);
  }
  scoreTransaction();
});
