let currentRun = null;
let baselineRun = null;
let runTraces = [];

function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatConfidence(confidence) {
    if (confidence === null || confidence === undefined || typeof confidence !== 'number' || isNaN(confidence)) {
        return 'Not calibrated';
    }
    return `${(confidence * 100).toFixed(0)}%`;
}

function formatMetric(metricObj, isPercentage = true) {
    if (!metricObj || metricObj.count === 0 || metricObj.mean === null || metricObj.mean === undefined || isNaN(metricObj.mean)) {
        return { display: '—', sub: 'No evaluation data', isEvaluated: false };
    }
    const display = isPercentage ? `${(metricObj.mean * 100).toFixed(1)}%` : String(metricObj.mean);
    return { display, sub: `N=${metricObj.count}`, isEvaluated: true };
}

async function init() {
    await fetchRuns();
}

async function fetchRuns() {
    try {
        const res = await fetch('/v1/runs');
        const data = await res.json();
        const runs = data.runs || [];
        const selector = document.getElementById('run-select');

        if (runs.length === 0) {
            renderEmptyState();
            return;
        }

        selector.innerHTML = runs.map((r, i) => {
            const safeSysVer = escapeHtml(r.system_version || 'unknown');
            const safeId = escapeHtml(r.id ? r.id.substring(0, 10) : 'unknown');
            const count = r.trace_count || 0;
            return `<option value="${escapeHtml(r.id)}" ${i === 0 ? 'selected' : ''}>${safeSysVer} (${safeId}) - ${count} cases</option>`;
        }).join('');

        currentRun = runs[0];
        baselineRun = runs.length > 1 ? runs[1] : null;

        renderRunDashboard(currentRun, baselineRun);
        await loadTraces(currentRun.id);
    } catch (err) {
        console.error("Failed to load runs:", err);
        renderEmptyState("Datastore unreachable. Ensure backend is running.");
    }
}

async function loadSelectedRun() {
    const selector = document.getElementById('run-select');
    if (!selector) return;
    const runId = selector.value;
    if (!runId) return;
    const res = await fetch(`/v1/runs/${encodeURIComponent(runId)}`);
    currentRun = await res.json();
    renderRunDashboard(currentRun, baselineRun);
    await loadTraces(runId);
}

async function loadTraces(runId, failureOnly = false) {
    try {
        const res = await fetch(`/v1/runs/${encodeURIComponent(runId)}/traces?failure_only=${failureOnly}`);
        const data = await res.json();
        runTraces = data.traces || [];
        renderTraceTable(runTraces);
    } catch(e) {
        console.error("Failed loading traces", e);
    }
}

function renderRunDashboard(run, baseline) {
    const root = document.getElementById('app-root');
    if (!root) return;
    const summary = run.summary || {};
    const metrics = summary.metrics || {};

    const faithMetric = formatMetric(metrics.faithfulness);
    const recallMetric = formatMetric(metrics.recall_at_5);
    const citMetric = formatMetric(metrics.citation_accuracy);

    let abstDisplay = '—';
    let abstSub = 'No evaluation data';
    if (summary.abstention_accuracy !== undefined && summary.abstention_accuracy !== null && !isNaN(summary.abstention_accuracy)) {
        abstDisplay = `${(summary.abstention_accuracy * 100).toFixed(1)}%`;
        abstSub = `N=${run.trace_count || 0}`;
    }

    const p95_lat = summary.p95_latency_ms !== undefined && summary.p95_latency_ms !== null ? summary.p95_latency_ms : '—';
    const cost = summary.total_cost_usd !== undefined && summary.total_cost_usd !== null ? `$${summary.total_cost_usd.toFixed(4)}` : '—';

    // Baseline deltas if available
    const bMetrics = baseline && baseline.summary ? baseline.summary.metrics || {} : {};
    const dFaith = (bMetrics.faithfulness && faithMetric.isEvaluated && bMetrics.faithfulness.mean !== null)
        ? (metrics.faithfulness.mean - bMetrics.faithfulness.mean) : null;
    const dRecall = (bMetrics.recall_at_5 && recallMetric.isEvaluated && bMetrics.recall_at_5.mean !== null)
        ? (metrics.recall_at_5.mean - bMetrics.recall_at_5.mean) : null;

    // Update Provenance Hash Badge
    const repro = document.getElementById('repro-badge');
    if (repro) {
        if (run.manifest_hash) {
            repro.style.display = 'inline-flex';
            repro.textContent = `SHA256: ${escapeHtml(run.manifest_hash.substring(0, 10))}`;
        } else {
            repro.style.display = 'none';
        }
    }

    // Backend regression/gate engine is the SINGLE source of truth (FIX #3)
    const gate = run.gate_result || null;
    const isPassed = gate ? (gate.status === 'PASS') : false;
    const gateStatus = gate ? escapeHtml(gate.status) : 'NO_DATA';
    const policyId = gate ? escapeHtml(gate.policy_id) : escapeHtml(run.policy_id || 'prod-default');
    const violations = (gate && Array.isArray(gate.violations)) ? gate.violations : [];

    root.innerHTML = `
        <!-- Executive Story Bar (Derived 100% from backend GateResult) -->
        <div class="story-bar">
            <div class="story-content">
                <h2>
                    ${isPassed ? '✓ Candidate Satisfies Release Guardrails' : '❌ Release Quality Gate Blocked'}
                </h2>
                <p>Policy: <code>${policyId}</code> | Evaluated ${escapeHtml(String(run.trace_count || 0))} test cases on benchmark <code>${escapeHtml(run.dataset_id || '')}</code> against commit <code>${escapeHtml(run.system_version || '')}</code>.</p>
                ${violations.length > 0 ? `
                    <div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:6px;">
                        ${violations.map(v => `<span class="chip chip-fail" style="font-size:11px;">${escapeHtml(v.metric_name)}: ${v.actual_value !== null ? escapeHtml(String(v.actual_value)) : 'No data'} (threshold ${escapeHtml(v.direction || '')} ${escapeHtml(String(v.threshold || ''))})</span>`).join('')}
                    </div>
                ` : ''}
            </div>
            <div class="story-status ${isPassed ? 'status-pass' : 'status-fail'}">
                GATE: ${gateStatus}
            </div>
        </div>

        <!-- Core Metric Tiles -->
        <div class="metric-grid">
            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">Claim Faithfulness</span>
                    <span class="chip ${faithMetric.isEvaluated && metrics.faithfulness.mean >= 0.85 ? 'chip-pass' : (faithMetric.isEvaluated ? 'chip-fail' : 'chip-neutral')}">
                        ${faithMetric.isEvaluated ? (metrics.faithfulness.mean >= 0.85 ? 'HEALTHY' : 'DRIFT') : 'UNSET'}
                    </span>
                </div>
                <div class="metric-val">${faithMetric.display}</div>
                <div class="metric-footer">
                    <span class="delta-tag ${dFaith && dFaith >= 0 ? 'delta-improved' : (dFaith !== null ? 'delta-regressed' : 'delta-neutral')}">
                        ${dFaith !== null ? (dFaith >= 0 ? '+' : '') + (dFaith * 100).toFixed(1) + '% vs base' : faithMetric.sub}
                    </span>
                    <span class="ci-span">${faithMetric.isEvaluated ? `95% CI: [${metrics.faithfulness.ci_lower || 0}, ${metrics.faithfulness.ci_upper || 1}]` : 'Not applicable'}</span>
                </div>
            </div>

            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">Evidence Recall@5</span>
                    <span class="chip ${recallMetric.isEvaluated && metrics.recall_at_5.mean >= 0.90 ? 'chip-pass' : (recallMetric.isEvaluated ? 'chip-warn' : 'chip-neutral')}">
                        ${recallMetric.isEvaluated ? 'RANKED' : 'UNSET'}
                    </span>
                </div>
                <div class="metric-val">${recallMetric.display}</div>
                <div class="metric-footer">
                    <span class="delta-tag ${dRecall && dRecall >= 0 ? 'delta-improved' : (dRecall !== null ? 'delta-regressed' : 'delta-neutral')}">
                        ${dRecall !== null ? (dRecall >= 0 ? '+' : '') + (dRecall * 100).toFixed(1) + '% vs base' : recallMetric.sub}
                    </span>
                    <span class="ci-span">${recallMetric.isEvaluated ? `95% CI: [${metrics.recall_at_5.ci_lower || 0}, ${metrics.recall_at_5.ci_upper || 1}]` : 'Not applicable'}</span>
                </div>
            </div>

            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">Citation Accuracy</span>
                    <span class="chip chip-neutral">${citMetric.isEvaluated ? 'VERIFIED' : 'UNSET'}</span>
                </div>
                <div class="metric-val">${citMetric.display}</div>
                <div class="metric-footer">
                    <span class="delta-tag delta-neutral">Evidence Links</span>
                    <span class="ci-span">${citMetric.sub}</span>
                </div>
            </div>

            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">Abstention Accuracy</span>
                    <span class="chip ${summary.abstention_accuracy !== undefined && summary.abstention_accuracy >= 0.90 ? 'chip-pass' : (summary.abstention_accuracy !== undefined ? 'chip-fail' : 'chip-neutral')}">BOUNDARY</span>
                </div>
                <div class="metric-val">${abstDisplay}</div>
                <div class="metric-footer">
                    <span class="delta-tag delta-neutral">Unanswerable Handling</span>
                    <span class="ci-span">${abstSub}</span>
                </div>
            </div>

            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">P95 Latency & Cost</span>
                    <span class="chip chip-neutral">${p95_lat !== '—' ? p95_lat + 'ms' : '—'}</span>
                </div>
                <div class="metric-val">${p95_lat !== '—' ? p95_lat + 'ms' : '—'}</div>
                <div class="metric-footer">
                    <span class="delta-tag delta-neutral">Cost: ${cost}</span>
                    <span class="ci-span">P95 Budget &le; 1200ms</span>
                </div>
            </div>
        </div>

        <!-- Trace Debugger & Deep Causal Analysis -->
        <div class="section-box">
            <div class="section-title">
                <h3>Evaluation Traces & Root Cause Inspector</h3>
                <div>
                    <button id="btn-filter-all" class="btn" onclick="filterTraces('all')">All Traces</button>
                    <button id="btn-filter-fail" class="btn" onclick="filterTraces('fail')">Failed Traces Only</button>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Trace ID</th>
                        <th>Question</th>
                        <th>Faithfulness</th>
                        <th>Recall</th>
                        <th>Primary Diagnosis</th>
                        <th>Contributing Causes</th>
                    </tr>
                </thead>
                <tbody id="trace-table-body">
                    <tr><td colspan="6" style="text-align:center; color:var(--text-dim);">Loading execution traces...</td></tr>
                </tbody>
            </table>
        </div>
    `;
}

function renderTraceTable(traces) {
    const tbody = document.getElementById('trace-table-body');
    if (!tbody) return;

    if (!Array.isArray(traces) || traces.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 24px; color: var(--text-dim);">No traces recorded for this evaluation run.</td></tr>`;
        return;
    }

    tbody.innerHTML = traces.map((t, idx) => {
        try {
            if (!t) return '';
            const traceId = t.trace_id ? String(t.trace_id) : 'trace_unknown';
            const shortId = traceId.length > 10 ? traceId.substring(0, 10) : traceId;
            const metrics = Array.isArray(t.metrics) ? t.metrics : [];
            const faith = metrics.find(m => m && m.name === 'faithfulness');
            const recall = metrics.find(m => m && m.name === 'recall_at_5');
            const fScore = (faith && faith.score !== null && faith.score !== undefined && !isNaN(faith.score)) ? faith.score.toFixed(2) : '—';
            const rScore = (recall && recall.score !== null && recall.score !== undefined && !isNaN(recall.score)) ? recall.score.toFixed(2) : '—';

            const fail = t.failure || null;
            const pCode = fail && fail.primary_code ? escapeHtml(fail.primary_code) : 'PASS';
            const contribList = (fail && Array.isArray(fail.contributing_codes)) ? fail.contributing_codes : [];
            const contrib = contribList.length > 0
                ? contribList.map(c => `<span class="chip chip-warn">${escapeHtml(c)}</span>`).join(' ')
                : '<span style="color:var(--text-dim);">-</span>';

            const statusChip = fail
                ? `<span class="chip chip-fail">${pCode}</span>`
                : `<span class="chip chip-pass">PASS</span>`;

            const chunks = (t.chunks && Array.isArray(t.chunks)) ? t.chunks : [];
            const citations = (t.citations && Array.isArray(t.citations)) ? t.citations : [];
            const actions = (fail && Array.isArray(fail.recommended_actions)) ? fail.recommended_actions : [];

            let answerText = '<em>Empty</em>';
            if (t.answer) {
                answerText = escapeHtml(t.answer);
            } else if (t.abstained) {
                answerText = `<em>Abstained: ${escapeHtml(t.abstention_reason || 'insufficient evidence')}</em>`;
            }

            const confText = fail ? formatConfidence(fail.confidence) : 'Not calibrated';

            return `
                <tr class="trace-row" onclick="toggleTraceDetail('tr-detail-${idx}')">
                    <td style="font-family:var(--font-mono); color:#38bdf8;">${escapeHtml(shortId)}</td>
                    <td>${escapeHtml(t.question || '')}</td>
                    <td style="font-family:var(--font-mono); color:${fScore !== '—' && parseFloat(fScore) >= 0.70 ? 'var(--pass)' : (fScore === '—' ? 'var(--text-dim)' : 'var(--fail)')};">${escapeHtml(fScore)}</td>
                    <td style="font-family:var(--font-mono);">${escapeHtml(rScore)}</td>
                    <td>${statusChip}</td>
                    <td>${contrib}</td>
                </tr>
                <tr id="tr-detail-${idx}" class="detail-row" style="display:none;">
                    <td colspan="6" class="detail-pane">
                        <div class="pipeline-stepper">
                            <!-- Step 1: Retrieval Chunks -->
                            <div class="step-card">
                                <div class="step-label">1. Retrieved Context Chunks</div>
                                <div class="step-body">
                                    ${chunks.length > 0 ? chunks.map(c =>
                                        `<div class="chunk-pill"><strong>[Rank ${escapeHtml(c.rank !== undefined ? c.rank : 1)}] ${escapeHtml(c.document_id || '')}:${escapeHtml(c.chunk_id || '')}</strong><br/>${escapeHtml(c.text ? c.text.substring(0, 140) : '')}...</div>`
                                    ).join('') : '<p style="color:var(--text-dim);">No context retrieved.</p>'}
                                </div>
                            </div>

                            <!-- Step 2: Generated Response -->
                            <div class="step-card">
                                <div class="step-label">2. Generated Answer & Citations</div>
                                <div class="step-body">
                                    <p style="margin-bottom:6px;">${answerText}</p>
                                    ${citations.length > 0 ? citations.map(c =>
                                        `<span class="chip chip-neutral" style="font-size:10px;">Cited: ${escapeHtml(c.document_id || '')}:${escapeHtml(c.chunk_id || '')}</span>`
                                    ).join(' ') : ''}
                                </div>
                            </div>

                            <!-- Step 3: Diagnostic Findings -->
                            <div class="step-card">
                                <div class="step-label">3. Root Cause Explanation</div>
                                <div class="step-body">
                                    ${fail ? `
                                        <p style="color:var(--fail); font-weight:600; margin-bottom:4px;">Primary: ${escapeHtml(fail.primary_code || '')} (Conf: ${escapeHtml(confText)})</p>
                                        <p style="font-size:11px; color:#d4d4d8;">${escapeHtml(fail.explanation || '')}</p>
                                    ` : '<p style="color:var(--pass);">✓ All claims verified against retrieved evidence.</p>'}
                                </div>
                            </div>

                            <!-- Step 4: Recommended Action -->
                            <div class="step-card">
                                <div class="step-label">4. Remediation Action</div>
                                <div class="step-body">
                                    ${actions.length > 0 ? `
                                        <ul style="padding-left:14px; color:var(--text-muted); font-size:11px;">
                                            ${actions.map(a => `<li>${escapeHtml(a)}</li>`).join('')}
                                        </ul>
                                    ` : '<p style="color:var(--text-dim);">No corrective action required.</p>'}
                                </div>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
        } catch (err) {
            console.warn("Skipping corrupt trace at index", idx, err);
            return `<tr class="trace-row"><td colspan="6" style="color:var(--fail); font-size:12px;">Trace ${idx} record unavailable (malformed data)</td></tr>`;
        }
    }).join('');
}

function toggleTraceDetail(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.display = (el.style.display === 'table-row') ? 'none' : 'table-row';
}

async function filterTraces(type) {
    if (!currentRun || !currentRun.id) return;
    const failureOnly = (type === 'fail');
    await loadTraces(currentRun.id, failureOnly);
}

function renderEmptyState(msg) {
    const root = document.getElementById('app-root');
    if (!root) return;
    root.innerHTML = `
        <div class="empty-state">
            <h4>No Benchmark Evaluation Runs Found</h4>
            <p>${escapeHtml(msg) || 'Initialize your first evaluation run via the CLI, REST API, or click below to bootstrap an evaluation run with live golden benchmarks.'}</p>
            <div class="cli-box">python -m rag_platform.gate --project prj-001 --dataset ds-gold --system-version git-abc123</div>
            <br/>
            <button class="btn btn-primary" onclick="triggerSeedRun()">Initialize Sample Benchmark Run</button>
        </div>
    `;
}

async function triggerSeedRun() {
    try {
        const res = await fetch('/v1/demo-run', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'SUCCESS') {
            await fetchRuns();
        } else {
            alert('Error creating benchmark run: ' + JSON.stringify(data));
        }
    } catch(e) {
        alert('Benchmark trigger failed: ' + e);
    }
}

function refreshDashboard() {
    fetchRuns();
}

if (typeof window !== 'undefined') {
    window.onload = init;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        escapeHtml,
        formatConfidence,
        formatMetric,
        renderTraceTable,
        renderRunDashboard,
    };
}
