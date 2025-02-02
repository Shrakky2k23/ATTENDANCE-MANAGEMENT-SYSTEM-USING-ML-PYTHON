// script.js
function runScript(scriptName) {
    fetch(`/run?script=${scriptName}`)
        .then(response => response.text())
        .then(data => {
            document.getElementById("output").innerText = data;
        })
        .catch(error => {
            document.getElementById("output").innerText = "Error running script!";
        });
}
