document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const resultDiv = document.getElementById('result');

    // Show loading
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `
        <div class="loading" style="display:block;">
            <div class="spinner"></div>
            <p style="font-size:1.2em; color:var(--accent-primary); font-family:'Outfit', sans-serif;">Analyzing patient data...</p>
        </div>`;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `
                <div style="text-align:center; padding:40px;">
                    <h2 style="color:var(--danger); font-size:2.5em; margin-bottom:15px; font-family:'Outfit', sans-serif;">⚠️ Error</h2>
                    <p style="font-size:1.2em; color:var(--text-secondary);">${data.error}</p>
                </div>`;
            return;
        }

        const isHigh = data.risk_class === 'high';

        const featuresHTML = data.top_features.map((f, i) => `
            <div class="feature-item">
                <strong>${i + 1}. ${f.name}</strong>
                <span class="feature-val" style="float:right;">${f.value.toFixed(2)}</span>
                <div class="bar-container" style="height:15px; margin-top:12px;">
                    <div class="bar-fill ${isHigh ? 'high-risk' : 'low-risk'}" style="width:${f.importance.toFixed(1)}%; font-size:0.7em;">
                        ${f.importance.toFixed(1)}%
                    </div>
                </div>
            </div>`).join('');

        const recommendationsHTML = data.recommendations.map(r =>
            `<li>${r}</li>`).join('');

        resultDiv.innerHTML = `
            <div class="result-header ${isHigh ? 'high-risk' : 'low-risk'}">
                <h2>${isHigh ? '⚠️ HIGH RISK DETECTED' : '✅ LOW RISK'}</h2>
                <div class="risk-badge ${data.risk_class}">
                    ${data.prediction}
                </div>
            </div>

            <div class="probability-bars">
                <h3 style="margin-bottom:20px; color:var(--text-primary); font-size:1.8em; font-family:'Outfit', sans-serif;">📊 Prediction Confidence</h3>

                <div class="probability-bar">
                    <label>🔴 High Risk Probability</label>
                    <div class="bar-container">
                        <div class="bar-fill high-risk" style="width:${data.high_risk_probability.toFixed(1)}%">
                            ${data.high_risk_probability.toFixed(1)}%
                        </div>
                    </div>
                </div>

                <div class="probability-bar">
                    <label>🟢 Low Risk Probability</label>
                    <div class="bar-container">
                        <div class="bar-fill low-risk" style="width:${data.low_risk_probability.toFixed(1)}%">
                            ${data.low_risk_probability.toFixed(1)}%
                        </div>
                    </div>
                </div>
            </div>

            <div class="top-features">
                <h3>📈 Top 5 Contributing Features</h3>
                ${featuresHTML}
            </div>

            <div class="recommendations">
                <h3>${isHigh ? '⚠️ Action Required' : '✅ Health Maintenance Tips'}</h3>
                <ul>${recommendationsHTML}</ul>
            </div>

            <div style="text-align:center; margin-top:40px;">
                <button onclick="resetForm()" class="btn-predict" 
                    style="background: linear-gradient(90deg, #1c2331, #2d3851); max-width:350px;">
                    🔄 New Prediction
                </button>
            </div>`;

    } catch (err) {
        resultDiv.innerHTML = `
            <div style="text-align:center; padding:40px;">
                <h2 style="color:var(--danger); font-size:2.5em; margin-bottom:15px; font-family:'Outfit', sans-serif;">⚠️ Connection Error</h2>
                <p style="color:var(--text-secondary); font-size:1.1em;">Could not connect to server. Please try again.</p>
            </div>`;
    }

    // Scroll to result
    resultDiv.scrollIntoView({ behavior: 'smooth' });
});

function resetForm() {
    document.getElementById('predictionForm').reset();
    const resultDiv = document.getElementById('result');
    resultDiv.style.opacity = '0';
    setTimeout(() => {
        resultDiv.style.display = 'none';
        resultDiv.style.opacity = '1';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 300);
}
