'use strict';

// When served via GitHub Pages the path is relative to index.html
const DATA_URL = 'data/lmeval-results.json';

const COLORS = [
  '#58a6ff', '#3fb950', '#f78166', '#d2a8ff',
  '#ffa657', '#79c0ff', '#56d364', '#ff7b72',
];

let allData = [];
let activeTab = 'configs';
let charts = {};

async function init() {
  setStatus('Loading results…');
  try {
    const resp = await fetch(DATA_URL + '?_=' + Date.now());
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    allData = await resp.json();
    const last = allData.length ? new Date(allData[allData.length - 1].timestamp).toLocaleString() : 'never';
    setStatus(`${allData.length} run(s) loaded — last updated ${last}`);
  } catch (e) {
    setStatus('Could not load results: ' + e.message);
    return;
  }
  render();
}

function setStatus(msg) {
  document.getElementById('status-bar').textContent = msg;
}

function render() {
  const runs = allData.filter(r => r.configs === activeTab);
  renderSummary(runs);
  renderCharts(runs);
}

// ── Summary table ─────────────────────────────────────────────────────────────

function renderSummary(runs) {
  const wrap = document.getElementById('summary-table-wrap');
  if (!runs.length) {
    wrap.innerHTML = '<p class="empty-msg">No runs yet for this config type.</p>';
    return;
  }

  const latest = {};
  for (const run of runs) {
    for (const r of run.results) {
      latest[r.test_config] = { run, result: r };
    }
  }

  const rows = Object.entries(latest).map(([cfg, { run, result }]) => {
    const badge = result.passed
      ? '<span class="badge badge-pass">PASSED</span>'
      : '<span class="badge badge-fail">FAILED</span>';
    const date = new Date(run.timestamp).toLocaleString();
    const recoveries = Object.entries(result.recovery_metrics)
      .map(([, m]) => `${m.Recovery}%`)
      .join(', ');
    return `<tr>
      <td>${cfg.split('/').pop()}</td>
      <td>${badge}</td>
      <td>${recoveries || '—'}</td>
      <td>${run.llm_compressor_ref}</td>
      <td>${run.test_label}</td>
      <td>${date}</td>
    </tr>`;
  });

  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Config</th><th>Last Status</th><th>Recovery (last run)</th>
          <th>Branch</th><th>Runner</th><th>Timestamp</th>
        </tr>
      </thead>
      <tbody>${rows.join('')}</tbody>
    </table>`;
}

// ── Charts ────────────────────────────────────────────────────────────────────

function renderCharts(runs) {
  Object.values(charts).forEach(c => c.destroy());
  charts = {};

  const grid = document.getElementById('charts-grid');
  grid.innerHTML = '';

  if (!runs.length) {
    grid.innerHTML = '<p class="empty-msg">No runs yet for this config type.</p>';
    return;
  }

  // Group entries by config name
  const byConfig = {};
  for (const run of runs) {
    for (const r of run.results) {
      const key = r.test_config.split('/').pop();
      (byConfig[key] = byConfig[key] || []).push({ run, result: r });
    }
  }

  Object.entries(byConfig).forEach(([cfg, entries], idx) => {
    const card = document.createElement('div');
    card.className = 'chart-card';

    const metricNames = [...new Set(entries.flatMap(e => Object.keys(e.result.recovery_metrics)))];

    card.innerHTML = `
      <h3>${cfg}</h3>
      <div class="chart-meta">${metricNames.join(' · ')}</div>
      <canvas id="chart-${idx}"></canvas>`;
    grid.appendChild(card);

    const labels = entries.map(e => new Date(e.run.timestamp).toLocaleDateString());

    const datasets = metricNames.flatMap((metric, mi) => {
      const color = COLORS[mi % COLORS.length];
      const recoveries = entries.map(e => {
        const m = e.result.recovery_metrics[metric];
        return m ? parseFloat(m.Recovery) : null;
      });
      const thresholds = entries.map(e => {
        const m = e.result.recovery_metrics[metric];
        return m ? parseFloat(m.Threshold) : null;
      });
      const pointColors = entries.map(e => {
        const m = e.result.recovery_metrics[metric];
        if (!m) return color;
        return parseFloat(m.Recovery) >= parseFloat(m.Threshold) ? '#3fb950' : '#f85149';
      });

      return [
        {
          label: `${metric} (recovery)`,
          data: recoveries,
          borderColor: color,
          backgroundColor: color + '22',
          pointBackgroundColor: pointColors,
          pointRadius: 5,
          tension: 0.3,
          fill: false,
        },
        {
          label: `${metric} (threshold)`,
          data: thresholds,
          borderColor: color,
          borderDash: [4, 4],
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        },
      ];
    });

    const ctx = document.getElementById(`chart-${idx}`).getContext('2d');
    charts[idx] = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#8b949e', font: { size: 11 } }, position: 'bottom' },
          tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}%` } },
        },
        scales: {
          x: { ticks: { color: '#8b949e', font: { size: 11 } }, grid: { color: '#21262d' } },
          y: {
            min: 85, max: 105,
            ticks: { color: '#8b949e', font: { size: 11 }, callback: v => v + '%' },
            grid: { color: '#21262d' },
          },
        },
      },
    });
  });
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeTab = btn.dataset.tab;
    render();
  });
});

init();
