// Load and visualize placement data
async function loadData() {
    try {
        const response = await fetch('../processed_data/summary_statistics.json');
        const data = await response.json();

        // Populate stats cards
        document.getElementById('totalRecords').textContent = data.total_records.toLocaleString();
        document.getElementById('totalCompanies').textContent = data.total_companies.toLocaleString();
        document.getElementById('avgCtc').textContent = data.avg_fte_ctc.toFixed(2);
        document.getElementById('maxCtc').textContent = data.max_fte_ctc.toFixed(2);

        // Populate insights
        document.getElementById('medianCtc').innerHTML = `<strong>₹${data.median_fte_ctc.toFixed(2)} LPA</strong> median compensation for full-time roles`;
        document.getElementById('totalPlacement').innerHTML = `<strong>${data.total_placements.toFixed(0)}</strong> recorded placement offers`;
        document.getElementById('avgCgpa').innerHTML = `<strong>${data.avg_cgpa_cutoff.toFixed(2)}/10</strong> average CGPA requirement`;

        // Render charts
        renderYearChart(data.records_by_year);
        renderRecruiterChart(data.top_10_recruiters);
        renderTierChart(data.records_by_tier);

    } catch (error) {
        console.error('Error loading data:', error);
        document.querySelector('.container').innerHTML = '<div class="loading">Error loading data. Please ensure the data file exists.</div>';
    }
}

function renderYearChart(yearData) {
    const chartContainer = document.getElementById('yearChart');
    const maxValue = Math.max(...Object.values(yearData));

    chartContainer.innerHTML = '';

    Object.entries(yearData).forEach(([year, count]) => {
        const barItem = document.createElement('div');
        barItem.className = 'bar-item';

        const percentage = (count / maxValue) * 100;

        barItem.innerHTML = `
            <div class="bar-label">${year}</div>
            <div class="bar-wrapper">
                <div class="bar-fill" style="width: ${percentage}%">
                    ${count}
                </div>
            </div>
        `;

        chartContainer.appendChild(barItem);
    });
}

function renderRecruiterChart(recruiterData) {
    const chartContainer = document.getElementById('recruiterChart');
    const maxValue = Math.max(...Object.values(recruiterData));

    chartContainer.innerHTML = '';

    Object.entries(recruiterData).forEach(([company, count]) => {
        const barItem = document.createElement('div');
        barItem.className = 'bar-item';

        const percentage = (count / maxValue) * 100;

        barItem.innerHTML = `
            <div class="bar-label">${company}</div>
            <div class="bar-wrapper">
                <div class="bar-fill" style="width: ${percentage}%">
                    ${count}
                </div>
            </div>
        `;

        chartContainer.appendChild(barItem);
    });
}

function renderTierChart(tierData) {
    const chartContainer = document.getElementById('tierChart');

    // Combine similar tiers and filter out 'nan'
    const consolidatedTiers = {};

    Object.entries(tierData).forEach(([tier, count]) => {
        if (tier === 'nan') return;

        let normalizedTier = tier;

        // Group internships together
        if (tier.includes('Internship')) {
            normalizedTier = 'Internship (All)';
        }
        // Normalize tier naming (Tier 1 and Tier-1 should be same)
        else if (tier.includes('Tier')) {
            normalizedTier = tier.replace(/Tier[\s-]*/, 'Tier ');
        }

        consolidatedTiers[normalizedTier] = (consolidatedTiers[normalizedTier] || 0) + count;
    });

    // Sort tiers logically
    const sortedTiers = Object.entries(consolidatedTiers).sort((a, b) => {
        const order = ['Dream', 'Tier 1', 'Tier 2', 'Tier 3', 'Internship (All)'];
        return order.indexOf(a[0]) - order.indexOf(b[0]);
    });

    const maxValue = Math.max(...sortedTiers.map(([, count]) => count));

    chartContainer.innerHTML = '';

    sortedTiers.forEach(([tier, count]) => {
        const barItem = document.createElement('div');
        barItem.className = 'bar-item';

        const percentage = (count / maxValue) * 100;

        barItem.innerHTML = `
            <div class="bar-label">${tier}</div>
            <div class="bar-wrapper">
                <div class="bar-fill" style="width: ${percentage}%">
                    ${count}
                </div>
            </div>
        `;

        chartContainer.appendChild(barItem);
    });
}

// Add smooth animations when page loads
window.addEventListener('load', () => {
    loadData();

    // Animate stats cards
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';

            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 50);
        }, index * 100);
    });
});
