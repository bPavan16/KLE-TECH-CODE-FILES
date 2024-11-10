document.getElementById('signInForm')?.addEventListener('submit', function (event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    if (username && password) {
        alert(`Welcome, ${username}!`);
    } else {
        alert('Please fill out all fields.');
    }
});

document.getElementById('feedbackForm')?.addEventListener('submit', function (event) {
    event.preventDefault();
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const comments = document.getElementById('comments').value;
    if (name && email && comments) {
        alert('Thank you for your feedback!');
    } else {
        alert('Please fill out all fields.');
    }
});
