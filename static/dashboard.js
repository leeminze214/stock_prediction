// File: static/dashboard.js
const refreshBtn = document.getElementById('refreshBtn');
refreshBtn.addEventListener('click', () => location.reload());

// Load and render chart
type = '1d';
async function loadChart(interval) {
    const resp = await fetch(`/api/bars?bars=${interval}`);
    const data = await resp.json();
    const labels = data.map(d => new Date(d.Datetime));
    const close = data.map(d => d.Close);
    const sma = data.map(d => d.SMA20);
    const ema = data.map(d => d.EMA20);
    const bbH = data.map(d => d.BB_High);
    const bbL = data.map(d => d.BB_Low);
    const vwap = data.map(d => d.VWAP);

    new Chart(document.getElementById('priceChart'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'Close', data: close },
                { label: 'SMA20', data: sma },
                { label: 'EMA20', data: ema },
                { label: 'BB High', data: bbH },
                { label: 'BB Low', data: bbL },
                { label: 'VWAP', data: vwap }
            ]
        },
        options: { scales: { x: { type: 'time' } } }
    });
}

loadChart(type);
setInterval(() => loadChart(type), 60000);