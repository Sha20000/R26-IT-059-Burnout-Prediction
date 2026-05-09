const rows = document.querySelectorAll(".student-row");
const selectedStudent = document.getElementById("selected-student");
const selectedRisk = document.getElementById("selected-risk");
const selectedOverride = document.getElementById("selected-override");
const selectedExplanation = document.getElementById("selected-explanation");
const selectedRecommendation = document.getElementById("selected-recommendation");

let modelChart;
let distributionChart;

const chartColors = ["#1c7c7b", "#f08a5d", "#1c2b36", "#7c8a95"];

function formatValue(value) {
  if (value === null || value === undefined || value === "None") {
    return "N/A";
  }
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return "N/A";
  }
  return parsed.toFixed(2);
}

function updateInsight(row) {
  const studentId = row.dataset.student;
  const finalRisk = row.dataset.finalRisk;
  const override = row.dataset.override === "True" ? "Yes" : "No";

  selectedStudent.textContent = studentId;
  selectedRisk.textContent = `Risk level: ${finalRisk}`;
  selectedOverride.textContent = `Override: ${override}`;
  selectedOverride.className = `badge risk-${finalRisk.toLowerCase()}`;
  selectedExplanation.textContent = row.dataset.explanation;
  selectedRecommendation.textContent = row.dataset.recommendation;

  const scores = [
    Number(row.dataset.academic),
    Number(row.dataset.emotion),
    Number(row.dataset.behavior),
    Number(row.dataset.engagement),
  ].map((value) => (Number.isNaN(value) ? 0 : value));

  if (!modelChart) {
    modelChart = new Chart(document.getElementById("modelChart"), {
      type: "bar",
      data: {
        labels: ["Academic", "Emotion", "Behavior", "Engagement"],
        datasets: [
          {
            label: "Model Scores",
            data: scores,
            backgroundColor: chartColors,
            borderRadius: 6,
          },
        ],
      },
      options: {
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
          },
        },
        plugins: {
          legend: { display: false },
        },
      },
    });
  } else {
    modelChart.data.datasets[0].data = scores;
    modelChart.update();
  }
}

function initDistributionChart() {
  distributionChart = new Chart(document.getElementById("distributionChart"), {
    type: "doughnut",
    data: {
      labels: ["Low", "Medium", "High"],
      datasets: [
        {
          data: [distributionData.low, distributionData.medium, distributionData.high],
          backgroundColor: ["#cfe9dd", "#f6e3c2", "#f4c6c4"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

rows.forEach((row) => {
  row.addEventListener("click", () => updateInsight(row));
});

if (rows.length > 0) {
  updateInsight(rows[0]);
}

initDistributionChart();
