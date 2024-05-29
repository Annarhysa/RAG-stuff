function askQuestion() {
    const question = document.getElementById('question').value;
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('answer').innerText = data.answer;
    })
    .catch(error => console.error('Error:', error));
}
