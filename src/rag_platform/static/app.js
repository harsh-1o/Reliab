let currentRun = null;
let baselineRun = null;
let runTraces = [];

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

        selector.innerHTML = runs.map((r, i) =>
            `<option value="${r.id}" ${i === 0 ? 'selected' : ''}>${r.system_version} (${r.id.substring(0, 10)}) - ${r.trace_count} cases</option>`
        ).join('');

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
    const runId = document.getElementById('run-select').value;
    if (!runId) return;
    const res = await fetch(`/v1/runs/${runId}`);
    currentRun = await res.json();
    renderRunDashboard(currentRun, baselineRun);
    await loadTraces(runId);
}

async function loadTraces(runId) {
    try {
        const res = await fetch(`/v1/runs/${runId}/traces`);
        const data = await res.json();
        runTraces = data.traces || [];
        renderTraceTable(runTraces);
    } catch(e) {
        console.error("Failed loading traces", e);
    }
}

function renderRunDashboard(run, baseline) {
    const root = document.getElementById('app-root');
    const summary = run.summary || {};
    const metrics = summary.metrics || {};

    const faith = metrics.faithfulness || { mean: 0.0, count: 0, std_dev: 0 };
    const recall = metrics.recall_at_5 || { mean: 0.0, count: 0, std_dev: 0 };
    const cit = metrics.citation_accuracy || { mean: 0.0, count: 0, std_dev: 0 };
    const abst = summary.abstention_accuracy !== undefined ? summary.abstention_accuracy : 1.0;
    const p95_lat = summary.p95_latency_ms || 0;
    const cost = summary.total_cost_usd || 0;
    const halluc_rate = summary.hallucination_rate || 0.0;

    // Baseline deltas if available
    const bMetrics = baseline && baseline.summary ? baseline.summary.metrics || {} : {};
    const dFaith = bMetrics.faithfulness ? (faith.mean - bMetrics.faithfulness.mean) : null;
    const dRecall = bMetrics.recall_at_5 ? (recall.mean - bMetrics.recall_at_5.mean) : null;

    // Update Provenance Hash Badge
    const repro = document.getElementById('repro-badge');
    if (run.manifest_hash) {
        repro.style.display = 'inline-flex';
        repro.textContent = `SHA256: ${run.manifest_hash.substring(0, 10)}`;
    }

    const isPassed = halluc_rate <= 0.05 && faith.mean >= 0.85;

    root.innerHTML = `
        <!-- Executive Story Bar -->
        <div class="story-bar">
            <div class="story-content">
                <h2>
                    ${isPassed ? '✓ Candidate Satisfies Release Guardrails' : '❌ Release Quality Gate Blocked'}
                </h2>
                <p>Evaluated ${run.trace_count} test cases on benchmark <code>${run.dataset_id}</code> against commit <code>${run.system_version}</code>.</p>
            </div>
            <div class="story-status ${isPassed ? 'status-pass' : 'status-fail'}">
                GATE: ${isPassed ? 'PASS' : 'BLOCKED'}
            </div>
        </div>

        <!-- Core Metric Tiles with 95% Confidence Intervals -->
        <div class="metric-grid">
            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">Claim Faithfulness</span>
                    <span class="chip ${faith.mean >= 0.85 ? 'chip-pass' : 'chip-fail'}">${faith.mean >= 0.85 ? 'HEALTHY' : 'DRIFT'}</span>
                </div>
                <div class="metric-val">${(faith.mean * 100).toFixed(1)}%</div>
                <div class="metric-footer">
                    <span class="delta-tag ${dFaith && dFaith >= 0 ? 'delta-improved' : 'delta-regressed'}">
                        ${dFaith !== null ? (dFaith >= 0 ? '+' : '') + (dFaith * 100).toFixed(1) + '% vs base' : 'N=' + faith.count}
                    </span>
                    <span class="ci-span">95% CI: [${faith.ci_lower || 0}, ${faith.ci_upper || 1}]</span>
                </div>
            </div>

            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">Evidence Recall@5</span>
                    <span class="chip ${recall.mean >= 0.90 ? 'chip-pass' : 'chip-warn'}">RANKED</span>
                </div>
                <div class="metric-val">${(recall.mean * 100).toFixed(1)}%</div>
                <div class="metric-footer">
                    <span class="delta-tag ${dRecall && dRecall >= 0 ? 'delta-improved' : 'delta-regressed'}">
                        ${dRecall !== null ? (dRecall >= 0 ? '+' : '') + (dRecall * 100).toFixed(1) + '% vs base' : 'N=' + recall.count}
                    </span>
                    <span class="ci-span">95% CI: [${recall.ci_lower || 0}, ${recall.ci_upper || 1}]</span>
                </div>
            </div>

            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">Citation Accuracy</span>
                    <span class="chip chip-neutral">VERIFIED</span>
                </div>
                <div class="metric-val">${(cit.mean * 100).toFixed(1)}%</div>
                <div class="metric-footer">
                    <span class="delta-tag delta-neutral">Evidence Links</span>
                    <span class="ci-span">N=${cit.count}</span>
                </div>
            </div>

            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">Abstention Accuracy</span>
                    <span class="chip ${abst >= 0.90 ? 'chip-pass' : 'chip-fail'}">BOUNDARY</span>
                </div>
                <div class="metric-val">${(abst * 100).toFixed(1)}%</div>
                <div class="metric-footer">
                    <span class="delta-tag delta-neutral">Unanswerable Handling</span>
                    <span class="ci-span">N=${run.trace_count}</span>
                </div>
            </div>

            <div class="metric-tile">
                <div class="metric-header">
                    <span class="metric-label">P95 Latency & Cost</span>
                    <span class="chip chip-neutral">${p95_lat}ms</span>
                </div>
                <div class="metric-val">${p95_lat}ms</div>
                <div class="metric-footer">
                    <span class="delta-tag delta-neutral">Cost: $${cost.toFixed(4)}</span>
                    <span class="ci-span">P95 Budget &le; 1200ms</span>
                </div>
            </div>
        </div>

        <!-- Trace Debugger & Deep Causal Analysis -->
        <div class="section-box">
            <div class="section-title">
                <h3>Evaluation Traces & Root Cause Inspector</h3>
                <div>
                    <button class="btn" onclick="filterTraces('all')">All Traces</button>
                    <button class="btn" onclick="filterTraces('fail')">Failed Traces Only</button>
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

    if (traces.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 24px; color: var(--text-dim);">No traces recorded for this evaluation run.</td></tr>`;
        return;
    }

    tbody.innerHTML = traces.map((t, idx) => {
        const faith = t.metrics.find(m => m.name === 'faithfulness');
        const recall = t.metrics.find(m => m.name === 'recall_at_5');
        const fScore = faith && faith.score !== null ? faith.score.toFixed(2) : '-';
        const rScore = recall && recall.score !== null ? recall.score.toFixed(2) : '-';

        const fail = t.failure;
        const pCode = fail ? fail.primary_code : 'PASS';
        const contrib = fail && fail.contributing_codes && fail.contributing_codes.length > 0
            ? fail.contributing_codes.map(c => `<span class="chip chip-warn">${c}</span>`).join(' ')
            : '<span style="color:var(--text-dim);">-</span>';

        const statusChip = fail
            ? `<span class="chip chip-fail">${pCode}</span>`
            : `<span class="chip chip-pass">PASS</span>`;

        return `
            <tr class="trace-row" onclick="toggleTraceDetail('tr-detail-${idx}')">
                <td style="font-family:var(--font-mono); color:#38bdf8;">${t.trace_id.substring(0, 10)}</td>
                <td>${t.question}</td>
                <td style="font-family:var(--font-mono); color:${fScore >= 0.70 ? 'var(--pass)' : 'var(--fail)'};">${fScore}</td>
                <td style="font-family:var(--font-mono);">${rScore}</td>
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
                                ${t.chunks && t.chunks.length > 0 ? t.chunks.map(c =>
                                    `<div class="chunk-pill"><strong>[Rank ${c.rank}] ${c.document_id}:${c.chunk_id}</strong><br/>${c.text.substring(0, 140)}...</div>`
                                ).join('') : '<p style="color:var(--text-dim);">No context retrieved.</p>'}
                            </div>
                        </div>

                        <!-- Step 2: Generated Response -->
                        <div class="step-card">
                            <div class="step-label">2. Generated Answer & Citations</div>
                            <div class="step-body">
                                <p style="margin-bottom:6px;">${t.answer || (t.abstained ? '<em>Abstained: ' + (t.abstention_reason || 'insufficient evidence') + '</em>' : '<em>Empty</em>')}</p>
                                ${t.citations && t.citations.length > 0 ? t.citations.map(c =>
                                    `<span class="chip chip-neutral" style="font-size:10px;">Cited: ${c.document_id}:${c.chunk_id}</span>`
                                ).join(' ') : ''}
                            </div>
                        </div>

                        <!-- Step 3: Diagnostic Findings -->
                        <div class="step-card">
                            <div class="step-label">3. Root Cause Explanation</div>
                            <div class="step-body">
                                ${fail ? `
                                    <p style="color:var(--fail); font-weight:600; margin-bottom:4px;">Primary: ${fail.primary_code} (Conf: ${(fail.confidence * 100).toFixed(0)}%)</p>
                                    <p style="font-size:11px; color:#d4d4d8;">${fail.explanation}</p>
                                ` : '<p style="color:var(--pass);">✓ All claims verified against retrieved evidence.</p>'}
                            </div>
                        </div>

                        <!-- Step 4: Recommended Action -->
                        <div class="step-card">
                            <div class="step-label">4. Remediation Action</div>
                            <div class="step-body">
                                ${fail && fail.recommended_actions && fail.recommended_actions.length > 0 ? `
                                    <ul style="padding-left:14px; color:var(--text-muted); font-size:11px;">
                                        ${fail.recommended_actions.map(a => `<li>${a}</li>`).join('')}
                                    </ul>
                                ` : '<p style="color:var(--text-dim);">No corrective action required.</p>'}
                            </div>
                        </div>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
}

function toggleTraceDetail(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.display = (el.style.display === 'table-row') ? 'none' : 'table-row';
}

function filterTraces(type) {
    if (type === 'fail') {
        const fails = runTraces.filter(t => t.failure !== null);
        renderTraceTable(fails);
    } else {
        renderTraceTable(runTraces);
    }
}

function renderEmptyState(msg) {
    const root = document.getElementById('app-root');
    root.innerHTML = `
        <div class="empty-state">
            <h4>No Benchmark Evaluation Runs Found</h4>
            <p>${msg || 'Initialize your first evaluation run via the CLI, REST API, or click below to bootstrap an evaluation run with live golden benchmarks.'}</p>
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

window.onload = init;
