const BACKEND = "http://localhost:5000";

const status = document.getElementById("status");
const result = document.getElementById("result");

window.onload = async function () {

  try {
    let tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    let url = tabs[0].url;

    let match = url.match(/v=([^&]+)/);

    if (!match) {
      status.innerHTML = "❌ Open a YouTube video";
      return;
    }

    let videoId = match[1];

    status.innerHTML = "Fetching comments...";

    let comments = await getComments(videoId);

    if (comments.length === 0) {
      status.innerHTML = "No comments found";
      return;
    }

    status.innerHTML = "Analyzing comments...";

    let predictions = await analyzeComments(comments);

    status.innerHTML = "";
    showResult(predictions);

  } catch (err) {
    console.error(err);
    status.innerHTML = "❌ Something went wrong. Check the console for details.";
  }
};

// Get comments via the Flask backend (keeps the YouTube API key server-side)
async function getComments(videoId) {

  let response = await fetch(BACKEND + "/fetch_comments", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ video_id: videoId })
  });

  if (!response.ok) {
    console.error("Failed to fetch comments:", await response.text());
    return [];
  }

  return await response.json();
}

// Send to Flask
async function analyzeComments(comments) {

  let response = await fetch(BACKEND + "/predict_with_timestamps", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ comments: comments })
  });

  if (!response.ok) {
    console.error("Failed to analyze comments:", await response.text());
    return [];
  }

  return await response.json();
}

function showResult(data) {

  let positive = 0;
  let negative = 0;
  let neutral = 0;

  data.forEach(item => {
    if (item.sentiment == "1") positive++;
    else if (item.sentiment == "-1") negative++;
    else neutral++;
  });

  let total = data.length;
  let score = (((positive - negative) / total) * 10).toFixed(2);

  result.innerHTML = `
    <div class="card">
      <h3>Summary</h3>
      <p>Total Comments: ${total}</p>
      <p class="positive">Positive: ${positive}</p>
      <p class="neutral">Neutral: ${neutral}</p>
      <p class="negative">Negative: ${negative}</p>
      <p>Sentiment Score: ${score}/10</p>
    </div>

    <h3>Top Comments</h3>
    ${data.slice(0, 10).map(item => `
      <div class="comment">
        ${item.comment}
        <br>
        <b>Sentiment: ${item.sentiment}</b>
      </div>
    `).join("")}
  `;

  loadChart({
    "1": positive,
    "0": neutral,
    "-1": negative
  });
}

async function loadChart(counts) {

  let response = await fetch(BACKEND + "/generate_chart", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ sentiment_counts: counts })
  });

  if (!response.ok) {
    console.error("Failed to generate chart:", await response.text());
    return;
  }

  let blob = await response.blob();
  let img = document.createElement("img");
  img.src = URL.createObjectURL(blob);

  result.appendChild(img);
}