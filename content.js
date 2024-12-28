(async () => {
    const reviews = document.querySelectorAll(".review-text-content");
    for (const review of reviews) {
      const text = review.innerText;
      const tone = await fetchToneAnalysis(text);
      const tag = document.createElement("span");
      tag.textContent = `[${tone}]`;
      tag.style.color = "blue";
      tag.style.marginLeft = "10px";
      review.parentNode.insertBefore(tag, review.nextSibling);
    }
  })();
  
  async function fetchToneAnalysis(text) {
    const response = await fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review: text })
    });
    const data = await response.json();
    return data.tone;
  }
  