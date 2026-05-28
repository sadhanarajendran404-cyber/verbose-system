document.addEventListener("DOMContentLoaded", async () => {
const $ = id => document.getElementById(id), charts = {};
const opts = await (await fetch("/api/options")).json();
["brand", "processor", "ram", "storage", "operating_system"].forEach(k => opts[k].forEach(o => $(k).add(new Option(o, o))));
const metrics = await (await fetch("/api/metrics")).json();
const best = Object.entries(metrics).reduce((a, b) => b[1].r2 > a[1].r2 ? b : a)[0];
$("metrics-container").innerHTML = Object.entries(metrics).map(([m, s]) => `
<div class="metric-item ${m === best ? 'best-model' : ''}">
${m === best ? '<div class="best-badge"><i class="fa-solid fa-crown"></i> Best Fit</div>' : ''}
<div class="metric-name"><i class="fa-solid ${m === 'SVR' ? 'fa-chart-line' : m === 'Decision Tree' ? 'fa-network-wired' : 'fa-tree'}"></i><span>${m} Regressor</span></div>
<div class="metric-values"><div>R² Score: <span>${s.r2.toFixed(4)}</span></div><div>RMSE: <span>₹${Math.round(s.rmse).toLocaleString("en-IN")}</span></div></div>
<div class="progress-bar-container"><div class="progress-bar" style="width: ${s.r2 * 100}%; background: ${m === best ? 'linear-gradient(90deg, #06b6d4, #22d3ee)' : 'linear-gradient(90deg, #6366f1, #818cf8)'}"></div></div>
</div>
`).join('');
// 3. Stats Charts
const stats = await (await fetch("/api/stats")).json();
const config = (type, labels, data, label, bg, border, extra={}) => ({
type, data: { labels, datasets: [{ label, data, backgroundColor: bg, borderColor: border, borderWidth: 1.5, ...extra }] },
options: {
responsive: true, maintainAspectRatio: false,
plugins: { legend: { labels: { color: '#9ca3af', font: { family: 'Outfit' } } } },
scales: type === 'doughnut' ? {} : {
x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af', font: { family: 'Outfit' } } },
y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af', font: { family: 'Outfit' } } }
}
}
});
const bCtx = $("brandChart").getContext("2d"), bGrad = bCtx.createLinearGradient(0, 0, 0, 300);
bGrad.addColorStop(0, 'rgba(99, 102, 241, 0.8)'); bGrad.addColorStop(1, 'rgba(6, 182, 212, 0.2)');
const rCtx = $("ramChart").getContext("2d"), rGrad = rCtx.createLinearGradient(0, 0, 0, 300);
rGrad.addColorStop(0, 'rgba(6, 182, 212, 0.3)'); rGrad.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
charts.brand = new Chart($("brandChart"), config('bar', stats.brand_prices.labels, stats.brand_prices.values, 'Average Price (₹)', bGrad, '#6366f1', { borderRadius: 8 }));
charts.ram = new Chart($("ramChart"), config('line', stats.ram_prices.labels, stats.ram_prices.values, 'Avg Price by RAM (₹)', rGrad, '#22d3ee', { borderWidth: 3, fill: true, tension: 0.3, pointBackgroundColor: '#22d3ee' }));
charts.processor = new Chart($("processorChart"), config('doughnut', stats.processor_counts.labels, stats.processor_counts.values, 'Processor Shares', ['#6366f1', '#4f46e5', '#3b82f6', '#06b6d4', '#14b8a6', '#0d9488'], 'rgba(21, 28, 48, 0.8)'));
document.querySelectorAll(".tab-btn").forEach(btn => btn.onclick = () => {
document.querySelectorAll(".tab-btn").forEach(t => t.classList.remove("active"));
btn.classList.add("active");
const type = btn.dataset.chart;
["brand", "ram", "processor"].forEach(t => $(`${t}-chart-wrapper`).classList.toggle("hidden", t !== type));
if (charts[type]) { charts[type].resize(); charts[type].update(); }
});
$("prediction-form").onsubmit = async (e) => {
e.preventDefault();
$("loading-overlay").classList.remove("hidden");
try {
const res = await fetch("/api/predict", {
method: "POST", headers: { "Content-Type": "application/json" },
body: JSON.stringify(Object.fromEntries(new FormData(e.target)))
});
const data = await res.json();
setTimeout(() => {
$("loading-overlay").classList.add("hidden");
$("prediction-placeholder").classList.add("hidden");
$("prediction-content").classList.remove("hidden");
let start = performance.now();
const tick = (now) => {
let progress = Math.min((now - start) / 1200, 1);
$("svr-price-val").innerText = Math.round((1 - Math.pow(1 - progress, 3)) * data.svr_price).toLocaleString("en-IN");
if (progress < 1) requestAnimationFrame(tick);
};
requestAnimationFrame(tick);
$("dt-price-val").innerText = "₹" + Math.round(data.dt_price).toLocaleString("en-IN");
$("rf-price-val").innerText = "₹" + Math.round(data.rf_price).toLocaleString("en-IN");
}, 600);
} catch {
alert("Error running prediction.");
$("loading-overlay").classList.add("hidden");
}
};
});
